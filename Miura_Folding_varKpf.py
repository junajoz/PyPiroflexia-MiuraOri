"""
Miura_Folding_varKpf.py
=======================
Simula un patrón Miura-ori con rigidez de pliegue (kpf) variable por pliegue.

Situación física modelada
--------------------------
En una lámina Miura fabricada con materiales compuestos o ensamblada por zonas,
no todos los pliegues tienen la misma rigidez. Este script permite asignar un
campo kpf_ij distinto a cada pliegue para estudiar:
  - Gradientes suaves (zona blanda → rígida)
  - Franjas rígidas / blandas alternadas
  - Zonas de actuación (pliegues con kpf muy baja = actuadores)
  - Defecto puntual de rigidez

Salidas generadas (iguales al pipeline original)
-------------------------------------------------
  1. Vista superior con mapa de color kpf por pliegue
  2. Curva desplazamiento vs. factor de carga (displacement.py)
  3. Gráficas energéticas (GraphPostProcess.py)
  4. Configuración final deformada (PlotFinalConfig.py)
  5. Comparación nominal vs. gradiente: curvas carga-desp. superpuestas
  6. Mapa 2D de kpf efectivo sobre la malla

Uso
---
Ajusta la sección "PARÁMETROS" y la función `build_kpf_field()` a tu caso.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

# ── Agrega el directorio del proyecto al path ─────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ConfigMiuraFromAB    import ConfigMiuraFromAB
from defect_fields        import build_nominal_fields
from PrepareData_varKpf   import PrepareData_varKpf
from Ogden                import Ogden
from PathAnalysis_varKpf  import PathAnalysis_varKpf as PathAnalysis
from EnhancedLinear       import EnhancedLinear
from PostProcess          import PostProcess
from displacement         import displacement
from GraphPostProcess     import GraphPostProcess
from PlotFinalConfig      import PlotFinalConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  PARÁMETROS  ── edita esta sección
# ═══════════════════════════════════════════════════════════════════════════════

sec_hor  = 5          # celdas horizontales
sec_vert = 5          # celdas verticales

theta  = 60           # ángulo de panel [°]
fdang  = 15           # ángulo de pliegue inicial [°]
a0     = 2.0          # longitud de panel a [u]
b0     = 2.0          # longitud de panel b [u]

# Rigidez base de pliegue y doblez
Kf_base = 1e-1
Kb      = Kf_base * 1

# Patrón de variación de kpf  ← CAMBIA AQUÍ para probar distintos casos
# Opciones implementadas en build_kpf_field():
#   'uniform'   → todos los pliegues con Kf_base  (reproducción del original)
#   'gradient'  → gradiente lineal izq→der   (Kf_base … Kf_base * factor_max)
#   'stripe'    → franjas alternas blanda/rígida
#   'soft_zone' → región central blanda
#   'actuator'  → columna central con kpf muy baja (actuador)
KPF_PATTERN = 'gradient'
KPF_FACTOR_MAX = 50.0      # relación máxima kpf_max / kpf_base para 'gradient'
KPF_STRIPE_RATIO = 5.0     # relación rígida/blanda para 'stripe'
KPF_SOFT_RATIO   = 1e-4   # fracción de kpf_base en zona blanda

E0   = 1e6
Abar = 1e-1

limlft = 0.1
limrht = 360 - 0.1

MaxIcr = 60
blam   = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def build_supp_load(sec_hor, sec_vert):
    """Condiciones de borde y carga idénticas al pipeline original."""
    leftx  = np.arange(0, (2 * sec_vert + 1))
    leftz  = np.arange(0, (2 * sec_vert + 1) + 1, 2)
    rightz = (np.arange(0, (2 * sec_vert + 1) + 1, 2)
              + (2 * sec_vert + 1) * (2 * sec_hor))

    Supp = np.array([
        [0, 0, 1, 0],
        *zip(leftx,  np.ones_like(leftx),  np.zeros_like(leftx),  np.zeros_like(leftx)),
        *zip(leftz,  np.zeros_like(leftz),  np.zeros_like(leftz),  np.ones_like(leftz)),
        *zip(rightz, np.zeros_like(rightz), np.zeros_like(rightz), np.ones_like(rightz)),
    ], dtype=float)

    indp = (np.arange(0, (sec_vert * 2 + 1))
            + (sec_vert * 2 + 1) * (sec_hor * 2))
    ff   = -np.ones(len(indp))
    Load = np.column_stack((indp, ff,
                            np.zeros_like(indp),
                            np.zeros_like(indp)))
    indp = Load[:, 0].astype(int)
    return Supp, Load, indp


def build_kpf_field(n_folds, Kf_base, pattern,
                    factor_max=10.0, stripe_ratio=5.0, soft_ratio=0.05):
    """
    Construye el array kpf de longitud n_folds según el patrón solicitado.

    Los pliegues están ordenados como los devuelve findfdbd(): primero los
    pliegues internos de izquierda a derecha y de abajo a arriba. El índice
    normalizado t ∈ [0,1] se usa como coordenada para los gradientes.

    Patrones disponibles
    --------------------
    'uniform'   : kpf_i = Kf_base  ∀ i
    'gradient'  : kpf crece linealmente de Kf_base a Kf_base * factor_max
    'stripe'    : alterna Kf_base (blanda) y Kf_base*stripe_ratio (rígida)
    'soft_zone' : zona central con kpf = Kf_base * soft_ratio
    'actuator'  : tercio central con kpf = Kf_base * soft_ratio (actuador)
    """
    t = np.linspace(0, 1, n_folds)   # coordenada normalizada por índice

    if pattern == 'uniform':
        kpf = np.full(n_folds, Kf_base)

    elif pattern == 'gradient':
        kpf = Kf_base * (1.0 + (factor_max - 1.0) * t)

    elif pattern == 'stripe':
        # Franjas cada ~10 pliegues
        period = max(10, n_folds // (sec_hor + sec_vert))
        kpf = np.where((np.arange(n_folds) // period) % 2 == 0,
                       Kf_base,
                       Kf_base * stripe_ratio)

    elif pattern == 'soft_zone':
        # Tercio central blando
        kpf = np.full(n_folds, Kf_base)
        i0  = n_folds // 3
        i1  = 2 * n_folds // 3
        kpf[i0:i1] = Kf_base * soft_ratio

    elif pattern == 'actuator':
        # Franja central estrecha = actuador
        kpf = np.full(n_folds, Kf_base)
        i0  = 2 * n_folds // 5
        i1  = 3 * n_folds // 5
        kpf[i0:i1] = Kf_base * soft_ratio

    else:
        raise ValueError(f"Patrón '{pattern}' no reconocido.")

    return kpf


def extract_total_energy(STAT):
    """Extrae la energía total final de STAT (igual que en los otros scripts)."""
    total_energy = np.nan
    if not isinstance(STAT, dict):
        return total_energy
    for key in ["PE", "TotalPE", "Energy", "Etot"]:
        if key not in STAT:
            continue
        value = STAT[key]
        if isinstance(value, dict):
            energy_terms = []
            for candidate in value.values():
                if isinstance(candidate, (np.ndarray, list, tuple)):
                    arr = np.asarray(candidate).ravel()
                    if arr.size > 0:
                        try:
                            energy_terms.append(float(arr[-1]))
                        except (TypeError, ValueError):
                            pass
            if energy_terms:
                total_energy = float(np.sum(energy_terms))
                break
        else:
            arr = np.asarray(value).ravel()
            if arr.size > 0:
                try:
                    total_energy = float(arr[-1])
                    break
                except (TypeError, ValueError):
                    continue
    return total_energy


def run_case(a_cells, b_cells, kpf,
             theta, fdang, Kb, E0, Abar,
             limlft, limrht, sec_hor, sec_vert,
             blam, MaxIcr):
    """Ejecuta un caso completo y devuelve todos los resultados."""
    Node, Panel, BDRY = ConfigMiuraFromAB(a_cells, b_cells, theta, fdang)

    BarMater  = lambda Ex: Ogden(Ex, E0)
    RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(
                    he, h0, kpi, L0, limlft, limrht)

    Supp, Load, indp = build_supp_load(sec_hor, sec_vert)

    truss, angles, F = PrepareData_varKpf(
        Node, Panel, Supp, Load,
        BarMater, RotSpring,
        kpf=kpf, kpb=Kb, Abar=Abar
    )
    truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

    U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
    U_his  = np.real(U_his)
    LF_his = np.real(LF_his)

    STAT = PostProcess(Data, truss, angles)

    return {
        'Node':         Node,
        'Panel':        Panel,
        'truss':        truss,
        'angles':       angles,
        'F':            F,
        'indp':         indp,
        'U_his':        U_his,
        'LF_his':       LF_his,
        'Data':         Data,
        'STAT':         STAT,
        'final_load':   float(LF_his[-1]) if len(LF_his) > 0 else np.nan,
        'max_abs_disp': float(np.max(np.abs(U_his))) if U_his.size > 0 else np.nan,
        'total_energy': extract_total_energy(STAT),
        'kpf':          np.asarray(kpf) if not np.isscalar(kpf)
                        else np.full(angles['fold'].shape[0], kpf),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIONES ESPECÍFICAS DE ESTE SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kpf_on_structure(Node, Panel, kpf_per_fold, Fold,
                          title="Mapa de rigidez kpf por pliegue"):
    """
    Vista superior de la malla con cada pliegue coloreado según su kpf.
    Un 'pliegue' está definido por los dos primeros nodos de cada fila de Fold.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Dibuja paneles en gris claro
    for panel in Panel:
        pts = Node[panel, :2]
        pts_c = np.vstack([pts, pts[0]])
        ax.fill(pts_c[:, 0], pts_c[:, 1],
                facecolor='#f0f0f0', edgecolor='#aaaaaa',
                linewidth=0.6, alpha=0.7)

    # Colorea cada pliegue (segmento entre nodo Fold[:,0] y Fold[:,1])
    norm  = Normalize(vmin=kpf_per_fold.min(), vmax=kpf_per_fold.max())
    cmap  = cm.plasma
    for idx, (fold_nodes, kval) in enumerate(zip(Fold, kpf_per_fold)):
        n0, n1 = fold_nodes[0], fold_nodes[1]
        x0, y0 = Node[n0, :2]
        x1, y1 = Node[n1, :2]
        color = cmap(norm(kval))
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=3.5, solid_capstyle='round')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='kpf [rigidez de pliegue]', shrink=0.85)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()


def plot_load_disp_comparison(nom, var, indp, label_var="Gradiente kpf"):
    """Superpone las curvas carga-desplazamiento de dos casos."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for case, label, color in [
        (nom, "Nominal (uniforme)", "#028090"),
        (var, label_var,           "#e574bc"),
    ]:
        dof = int(np.abs(indp[0]) * 3)
        dsp = case['U_his'][dof, :]
        ax.plot(dsp, case['LF_his'],
                color=color, linewidth=2, label=label)

    ax.set_xlabel("Desplazamiento [u]", fontsize=13)
    ax.set_ylabel("Factor de carga λ",  fontsize=13)
    ax.set_title("Comparación: nominal vs. kpf variable", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_energy_comparison(nom, var, label_var="Gradiente kpf"):
    """Energía total acumulada vs. incremento para los dos casos."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for case, label, color in [
        (nom, "Nominal (uniforme)", "#028090"),
        (var, label_var,           "#e574bc"),
    ]:
        pe = case['STAT']['PE']['strain']
        ax.plot(pe, color=color, linewidth=2, label=label)

    ax.set_xlabel("Incremento de carga", fontsize=13)
    ax.set_ylabel("Energía total [PE]",  fontsize=13)
    ax.set_title("Comparación de energía almacenada", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def print_summary(nom, var, label_var):
    """Imprime tabla comparativa de métricas globales."""
    print("\n" + "═" * 52)
    print(f"{'MÉTRICA':<28} {'NOMINAL':>10} {'VAR kpf':>10}")
    print("─" * 52)
    print(f"{'final_load':<28} {nom['final_load']:>10.4f} {var['final_load']:>10.4f}")
    print(f"{'max_abs_disp':<28} {nom['max_abs_disp']:>10.4f} {var['max_abs_disp']:>10.4f}")
    print(f"{'total_energy':<28} {nom['total_energy']:>10.4f} {var['total_energy']:>10.4f}")
    delta_load   = var['final_load']   - nom['final_load']
    delta_disp   = var['max_abs_disp'] - nom['max_abs_disp']
    delta_energy = var['total_energy'] - nom['total_energy']
    print("─" * 52)
    print(f"{'Δ final_load':<28} {delta_load:>+10.4f}")
    print(f"{'Δ max_abs_disp':<28} {delta_disp:>+10.4f}")
    print(f"{'Δ total_energy':<28} {delta_energy:>+10.4f}")
    print("═" * 52)
    print(f"Patrón kpf: {label_var}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Geometría nominal (a y b uniformes) ───────────────────────────────────
    a_cells, b_cells = build_nominal_fields(sec_hor, sec_vert, a0, b0)

    # ── Caso 1: kpf uniforme (nominal) ────────────────────────────────────────
    print("\n▶  Corriendo caso NOMINAL (kpf uniforme)…")
    kpf_uniform = Kf_base   # escalar → PrepareData_varKpf lo expande

    nominal = run_case(
        a_cells, b_cells, kpf_uniform,
        theta, fdang, Kb, E0, Abar,
        limlft, limrht, sec_hor, sec_vert,
        blam, MaxIcr
    )

    # ── Construir campo kpf variable ──────────────────────────────────────────
    # Necesitamos saber cuántos pliegues hay → corremos la geometría una vez
    _Node_tmp, _Panel_tmp, _ = ConfigMiuraFromAB(a_cells, b_cells, theta, fdang)
    # Importamos las funciones de topología para contar pliegues
    from findbend import findbend
    from findfdbd import findfdbd
    _Bend_tmp = findbend(_Panel_tmp, _Node_tmp)
    _Fold_tmp, _, _ = findfdbd(_Panel_tmp, _Bend_tmp)
    n_folds = _Fold_tmp.shape[0]
    print(f"   → Número de pliegues (folds) detectados: {n_folds}")

    kpf_var = build_kpf_field(
        n_folds, Kf_base, KPF_PATTERN,
        factor_max=KPF_FACTOR_MAX,
        stripe_ratio=KPF_STRIPE_RATIO,
        soft_ratio=KPF_SOFT_RATIO,
    )
    label_var = f"kpf variable — patrón '{KPF_PATTERN}'"

    # ── Caso 2: kpf variable ──────────────────────────────────────────────────
    print(f"\n▶  Corriendo caso kpf VARIABLE (patrón: {KPF_PATTERN})…")
    variante = run_case(
        a_cells, b_cells, kpf_var,
        theta, fdang, Kb, E0, Abar,
        limlft, limrht, sec_hor, sec_vert,
        blam, MaxIcr
    )

    # ── Tabla de métricas ─────────────────────────────────────────────────────
    print_summary(nominal, variante, label_var)

    # ══════════════════════════════════════════════════════════════════════════
    #  GRÁFICAS
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Mapa kpf sobre la malla (caso variable)
    plot_kpf_on_structure(
        variante['Node'],
        variante['Panel'],
        variante['kpf'],
        variante['angles']['fold'],
        title=f"Distribución de rigidez de pliegue — patrón '{KPF_PATTERN}'"
    )

    # 2. Curvas carga–desplazamiento superpuestas
    plot_load_disp_comparison(nominal, variante, nominal['indp'], label_var)

    # 3. Comparación de energía almacenada
    plot_energy_comparison(nominal, variante, label_var)

    # 4. Gráficas del pipeline original — caso NOMINAL
    print("\n▶  Gráficas del pipeline — caso nominal:")
    instdof = -(nominal['indp'][0] * 3)
    displacement(nominal['U_his'], instdof, nominal['LF_his'])
    plt.title("Desplazamiento vs. carga — Nominal")

    GraphPostProcess(nominal['U_his'], nominal['STAT'])
    plt.gcf().suptitle("Postproceso energético — Nominal")

    PlotFinalConfig(nominal['U_his'], nominal['truss'],
                    nominal['angles'], nominal['LF_his'])
    plt.title("Configuración final — Nominal")

    # 5. Gráficas del pipeline original — caso VARIABLE kpf
    print(f"\n▶  Gráficas del pipeline — caso {label_var}:")
    instdof_var = -(variante['indp'][0] * 3)
    displacement(variante['U_his'], instdof_var, variante['LF_his'])
    plt.title(f"Desplazamiento vs. carga — {label_var}")

    GraphPostProcess(variante['U_his'], variante['STAT'])
    plt.gcf().suptitle(f"Postproceso energético — {label_var}")

    PlotFinalConfig(variante['U_his'], variante['truss'],
                    variante['angles'], variante['LF_his'])
    plt.title(f"Configuración final — {label_var}")

    plt.show()
