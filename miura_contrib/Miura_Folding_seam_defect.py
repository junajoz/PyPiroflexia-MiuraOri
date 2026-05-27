"""
Miura_Folding_seam_defect_small.py
==================================
Versión compacta y más legible del estudio de costuras defectuosas en un
patrón Miura.

Mejoras respecto al script original
-----------------------------------
1. Usa una subestructura más pequeña para reducir tiempo de cómputo.
2. Usa un DOF monitor útil (componente X del nodo central cargado).
3. Grafica desplazamiento relativo para que las curvas F–δ sí se aprecien.
4. Mantiene el barrido por posición y severidad.
5. Conserva el pipeline completo para el peor caso.

Interpretación física
---------------------
Una costura defectuosa se modela como una fila o columna de celdas cuya
geometría local está degradada (a, b o ambas reducidas). Esto representa una
franja de unión mal fabricada que altera la respuesta global de la lámina.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ConfigMiuraFromAB  import ConfigMiuraFromAB
from defect_fields      import (
    build_nominal_fields,
    apply_row_defect,
    apply_col_defect
)
from PrepareData        import PrepareData
from PathAnalysis       import PathAnalysis
from PostProcess        import PostProcess
from Ogden              import Ogden
from EnhancedLinear     import EnhancedLinear
from displacement       import displacement
from GraphPostProcess   import GraphPostProcess
from PlotFinalConfig    import PlotFinalConfig


# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════
# Subestructura pequeña para barridos rápidos
sec_hor  = 3
sec_vert = 3

theta    = 60
fdang    = 15
a0       = 2.0
b0       = 2.0

Kf   = 1e-1
Kb   = Kf * 1e5
E0   = 1e6
Abar = 1e-1
limlft = 0.1
limrht = 360 - 0.1
MaxIcr = 50
blam   = 0.5

DEFECT_TYPE = "ab"      # 'a', 'b', o 'ab'
SEAM_MODE   = "row"     # 'row' o 'col'

N_SEV       = 4
SEV_MAX     = 0.40      # 40% de reducción máxima
FIXED_POS   = 1         # posición fija para curvas por severidad
FIXED_SEV   = 0.20      # severidad fija para curvas por posición


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_supports_and_load(sec_hor, sec_vert):
    """
    Soportes y carga equivalentes al caso base.
    La carga se aplica en X sobre el borde derecho.
    """
    n_row  = 2 * sec_vert + 1
    n_col  = 2 * sec_hor + 1
    leftx  = np.arange(n_row)
    leftz  = np.arange(0, n_row + 1, 2)
    rightz = np.arange(0, n_row + 1, 2) + n_row * (n_col - 1)

    Supp = np.array([
        [0, 0, 1, 0],
        *zip(leftx,  np.ones_like(leftx),  np.zeros_like(leftx),  np.zeros_like(leftx)),
        *zip(leftz,  np.zeros_like(leftz), np.zeros_like(leftz),  np.ones_like(leftz)),
        *zip(rightz, np.zeros_like(rightz), np.zeros_like(rightz), np.ones_like(rightz)),
    ], dtype=float)

    indp = np.arange(n_row) + n_row * (n_col - 1)
    ff   = -np.ones(len(indp))
    Load = np.column_stack((
        indp,
        ff,
        np.zeros_like(indp),
        np.zeros_like(indp)
    ))
    indp = Load[:, 0].astype(int)
    return Supp, Load, indp


def choose_monitor_node(indp):
    """
    Nodo monitor = nodo central del borde cargado.
    Mucho mejor que usar el primero.
    """
    return int(indp[len(indp) // 2])


def get_monitor_dof(indp):
    """
    Como la carga está aplicada en X, el DOF útil de monitoreo es X.
    """
    node = choose_monitor_node(indp)
    dof = node * 3 + 0
    return node, dof


def run_case(a_cells, b_cells):
    Node, Panel, BDRY = ConfigMiuraFromAB(a_cells, b_cells, theta, fdang)
    Supp, Load, indp  = make_supports_and_load(sec_hor, sec_vert)

    BarMater  = lambda Ex: Ogden(Ex, E0)
    RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(
        he, h0, kpi, L0, limlft, limrht
    )

    truss, angles, F = PrepareData(
        Node, Panel, Supp, Load, BarMater, RotSpring, Kf, Kb, Abar
    )
    truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

    U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
    U_his  = np.real(U_his)
    LF_his = np.real(LF_his)
    STAT   = PostProcess(Data, truss, angles)

    node_monitor, dof_monitor = get_monitor_dof(indp)

    return {
        'Node': Node,
        'Panel': Panel,
        'truss': truss,
        'angles': angles,
        'indp': indp,
        'monitor_node': node_monitor,
        'monitor_dof': dof_monitor,
        'U_his': U_his,
        'LF_his': LF_his,
        'STAT': STAT,
        'final_load': float(LF_his[-1]) if len(LF_his) > 0 else np.nan,
        'max_abs_disp': float(np.max(np.abs(U_his))) if U_his.size > 0 else np.nan,
    }


def apply_seam(a_nom, b_nom, pos, severity, mode, defect_type):
    """
    Aplica defecto de costura horizontal o vertical.
    """
    if mode == "row":
        return apply_row_defect(
            a_nom, b_nom, pos,
            defect_type=defect_type,
            severity=severity
        )
    return apply_col_defect(
        a_nom, b_nom, pos,
        defect_type=defect_type,
        severity=severity
    )


def n_positions(mode):
    return sec_vert if mode == "row" else sec_hor


def get_disp_relative(res):
    """
    Desplazamiento relativo en el DOF monitor para visualizar mejor las curvas.
    """
    dof = res['monitor_dof']
    dsp = res['U_his'][dof, :]
    return dsp - dsp[0]


def extract_pe_final(res):
    pe = res['STAT']['PE']['strain']
    return float(pe[-1]) if len(pe) > 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_seam_on_structure(Node, Panel, pos, mode, sec_hor, sec_vert, title):
    """
    Marca en rojo la franja de paneles afectada por la costura.
    """
    seam_panels = set()

    if mode == "row":
        row_panel = 2 * pos
        for dr in range(2):
            for cp in range(2 * sec_hor):
                pid = (row_panel + dr) * (2 * sec_hor) + cp
                seam_panels.add(pid)
    else:
        col_panel = 2 * pos
        for rp in range(2 * sec_vert):
            for dc in range(2):
                pid = rp * (2 * sec_hor) + (col_panel + dc)
                seam_panels.add(pid)

    fig, ax = plt.subplots(figsize=(7, 6))
    patches_n, patches_s = [], []

    for pid, panel in enumerate(Panel):
        pts = Node[panel, :2]
        poly = Polygon(pts, closed=True)
        if pid in seam_panels:
            patches_s.append(poly)
        else:
            patches_n.append(poly)

    if patches_n:
        ax.add_collection(PatchCollection(
            patches_n,
            facecolor='#dddddd',
            edgecolor='#888888',
            lw=0.6,
            alpha=0.85
        ))

    if patches_s:
        ax.add_collection(PatchCollection(
            patches_s,
            facecolor='#e63946',
            edgecolor='k',
            lw=1.0,
            alpha=0.9
        ))

    ax.scatter(Node[:, 0], Node[:, 1], s=8, color='k')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()


def plot_sensitivity_map(delta_matrix, severities, title, mode):
    """
    Mapa 2D: Δλ_final vs posición y severidad.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos_label = "Fila de costura" if mode == "row" else "Columna de costura"

    im = ax.imshow(
        delta_matrix,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        extent=[severities[0] * 100, severities[-1] * 100,
                -0.5, delta_matrix.shape[0] - 0.5]
    )
    ax.set_xlabel("Severidad del defecto [%]", fontsize=11)
    ax.set_ylabel(pos_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_yticks(np.arange(delta_matrix.shape[0]))
    plt.colorbar(im, ax=ax, label='Δλ_final (defectuoso − nominal)', shrink=0.85)
    plt.tight_layout()


def plot_curves_by_severity(cases_sev, nom_res, severities, pos):
    """
    Curvas F–δ para una posición fija y severidad variable.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = cm.plasma
    norm = Normalize(vmin=0, vmax=len(severities) - 1)

    dsp_nom = get_disp_relative(nom_res)
    ax.plot(dsp_nom, nom_res['LF_his'], color='k', lw=2, ls='--', label='Nominal')

    for k, (sev, res) in enumerate(zip(severities, cases_sev)):
        color = cmap(norm(k))
        dsp = get_disp_relative(res)
        ax.plot(dsp, res['LF_his'], color=color, lw=1.6, label=f"sev={sev*100:.0f}%")

    ax.set_xlabel(f"Desplazamiento relativo X [u]", fontsize=11)
    ax.set_ylabel("λ", fontsize=11)
    ax.set_title(f"Curvas F–δ: costura en posición {pos} (severidad variable)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_curves_by_position(cases_pos, nom_res, positions, sev):
    """
    Curvas F–δ para una severidad fija y posición variable.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=len(positions) - 1)

    dsp_nom = get_disp_relative(nom_res)
    ax.plot(dsp_nom, nom_res['LF_his'], color='k', lw=2, ls='--', label='Nominal')

    for k, (pos, res) in enumerate(zip(positions, cases_pos)):
        color = cmap(norm(k))
        dsp = get_disp_relative(res)
        ax.plot(dsp, res['LF_his'], color=color, lw=1.6, label=f"pos={pos}")

    ax.set_xlabel(f"Desplazamiento relativo X [u]", fontsize=11)
    ax.set_ylabel("λ", fontsize=11)
    ax.set_title(f"Curvas F–δ: severidad {sev*100:.0f}% (posición variable)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def print_summary(nom, worst, worst_pos, worst_sev):
    print("\n" + "═" * 64)
    print("Resumen del caso nominal y del peor caso")
    print("─" * 64)
    print(f"{'Caso nominal λ_final':<34} {nom['final_load']:>10.4f}")
    print(f"{'Peor caso λ_final':<34} {worst['final_load']:>10.4f}")
    print(f"{'Δλ_final':<34} {worst['final_load'] - nom['final_load']:>+10.4f}")
    print(f"{'Caso nominal |U|_max':<34} {nom['max_abs_disp']:>10.4f}")
    print(f"{'Peor caso |U|_max':<34} {worst['max_abs_disp']:>10.4f}")
    print(f"{'Caso nominal PE_final':<34} {extract_pe_final(nom):>10.4f}")
    print(f"{'Peor caso PE_final':<34} {extract_pe_final(worst):>10.4f}")
    print("─" * 64)
    print(f"{'Posición peor costura':<34} {worst_pos}")
    print(f"{'Severidad peor costura':<34} {worst_sev * 100:.0f}%")
    print(f"{'Nodo monitor':<34} {worst['monitor_node']}")
    print("═" * 64)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    a_nom, b_nom = build_nominal_fields(sec_hor, sec_vert, a0, b0)
    n_pos        = n_positions(SEAM_MODE)
    severities   = np.linspace(0.05, SEV_MAX, N_SEV)
    positions    = np.arange(n_pos)

    # ── Caso nominal ──────────────────────────────────────────────────────────
    print("\n▶ Caso NOMINAL")
    nom = run_case(a_nom, b_nom)
    print(f"   λ_final nominal = {nom['final_load']:.4f}")
    print(f"   nodo monitor = {nom['monitor_node']}, dof = {nom['monitor_dof']}")

    # ── Barrido posición × severidad ──────────────────────────────────────────
    delta_map = np.zeros((n_pos, N_SEV))

    print(f"\n▶ Barrido: {n_pos} posiciones × {N_SEV} severidades")
    for i, pos in enumerate(positions):
        for j, sev in enumerate(severities):
            a_def, b_def = apply_seam(a_nom, b_nom, pos, sev, SEAM_MODE, DEFECT_TYPE)
            res = run_case(a_def, b_def)
            delta_map[i, j] = res['final_load'] - nom['final_load']
            print(f"   pos={pos}, sev={sev:.2f} → Δλ = {delta_map[i, j]:+.4f}")

    plot_sensitivity_map(
        delta_map,
        severities,
        f"Sensibilidad Δλ — costura tipo '{SEAM_MODE}', defecto '{DEFECT_TYPE}'",
        SEAM_MODE
    )

    # ── Curvas por severidad ──────────────────────────────────────────────────
    fixed_pos = min(FIXED_POS, n_pos - 1)
    print(f"\n▶ Curvas por severidad — posición fija {fixed_pos}")
    cases_sev = []
    for sev in severities:
        a_def, b_def = apply_seam(a_nom, b_nom, fixed_pos, sev, SEAM_MODE, DEFECT_TYPE)
        cases_sev.append(run_case(a_def, b_def))

    plot_curves_by_severity(cases_sev, nom, severities, fixed_pos)

    # ── Curvas por posición ───────────────────────────────────────────────────
    print(f"\n▶ Curvas por posición — severidad fija {FIXED_SEV * 100:.0f}%")
    cases_pos = []
    for pos in positions:
        a_def, b_def = apply_seam(a_nom, b_nom, pos, FIXED_SEV, SEAM_MODE, DEFECT_TYPE)
        cases_pos.append(run_case(a_def, b_def))

    plot_curves_by_position(cases_pos, nom, positions, FIXED_SEV)

    # ── Peor caso ─────────────────────────────────────────────────────────────
    worst_idx = np.unravel_index(np.argmin(delta_map), delta_map.shape)
    worst_pos = int(positions[worst_idx[0]])
    worst_sev = float(severities[worst_idx[1]])

    print(f"\n▶ Peor caso: pos={worst_pos}, sev={worst_sev * 100:.0f}% "
          f"Δλ={delta_map[worst_idx]:+.4f}")

    a_worst, b_worst = apply_seam(a_nom, b_nom, worst_pos, worst_sev, SEAM_MODE, DEFECT_TYPE)
    worst_res = run_case(a_worst, b_worst)

    print_summary(nom, worst_res, worst_pos, worst_sev)

    # ── Visualización de la costura peor ──────────────────────────────────────
    plot_seam_on_structure(
        worst_res['Node'],
        worst_res['Panel'],
        worst_pos,
        SEAM_MODE,
        sec_hor,
        sec_vert,
        f"Peor costura marcada: {SEAM_MODE}={worst_pos}, sev={worst_sev*100:.0f}%"
    )

    # ── Pipeline del peor caso ────────────────────────────────────────────────
    try:
        dof_worst = -(worst_res['monitor_dof'])
        displacement(worst_res['U_his'], dof_worst, worst_res['LF_his'])
        plt.title(f"Despl. vs. carga — peor costura ({SEAM_MODE}={worst_pos})")
    except Exception as e:
        print(f"displacement: {e}")

    try:
        GraphPostProcess(worst_res['U_his'], worst_res['STAT'])
        plt.gcf().suptitle(f"Energía — peor costura ({SEAM_MODE}={worst_pos})")
    except Exception as e:
        print(f"GraphPostProcess: {e}")

    try:
        PlotFinalConfig(
            worst_res['U_his'],
            worst_res['truss'],
            worst_res['angles'],
            worst_res['LF_his']
        )
        plt.title("Config. final — peor costura")
    except Exception as e:
        print(f"PlotFinalConfig: {e}")

    plt.show()