"""
Miura_Folding_distributed_load.py
==================================
Numeral 3 — Comparación entre distintos esquemas de carga nodal en una
superficie Miura.

CASO A — 'right_edge'
    Igual al original del repo: carga vertical sobre el borde derecho.

CASO B — 'all_top'
    Carga vertical uniformemente repartida sobre los nodos elevados
    (crestas). No representa una presión continua uniforme sobre toda la
    superficie, sino una carga nodal concentrada sobre las líneas elevadas.

CASO C — 'weighted'
    Carga vertical repartida por área tributaria proyectada (en planta XY),
    aplicada solo en nodos libres en Z. Representa una distribución nodal
    equivalente vertical, no una presión normal local.

Nota metodológica
-----------------
La comparación F–δ entre casos se hace usando el MISMO nodo monitor global,
para evitar sesgos por escoger nodos distintos o restringidos.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ConfigMiura         import ConfigMiura
from PrepareData         import PrepareData
from PathAnalysis        import PathAnalysis
from PostProcess         import PostProcess
from Ogden               import Ogden
from EnhancedLinear      import EnhancedLinear
from displacement        import displacement
from GraphPostProcess    import GraphPostProcess
from PlotFinalConfig     import PlotFinalConfig


# ═══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════
sec_hor  = 5
sec_vert = 5
theta    = 60
a        = 2.0
b        = 2.0
fdang    = 15

Kf   = 1e-1
Kb   = Kf * 1e5
E0   = 1e6
Abar = 1e-1

limlft = 0.1
limrht = 360 - 0.1

MaxIcr = 60
blam   = 0.5

perturb_scale = 0.005
seed          = 42


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES DE GEOMETRÍA / APOYO
# ═══════════════════════════════════════════════════════════════════════════════

def make_supports(sec_hor, sec_vert):
    """Soportes idénticos al caso base del repo."""
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
    return Supp


def get_z_restrained_nodes(sec_hor, sec_vert):
    """
    Devuelve los nodos con restricción en z según make_supports.
    """
    n_row  = 2 * sec_vert + 1
    n_col  = 2 * sec_hor + 1
    leftz  = np.arange(0, n_row + 1, 2)
    rightz = np.arange(0, n_row + 1, 2) + n_row * (n_col - 1)
    return np.unique(np.concatenate([leftz, rightz])).astype(int)


def choose_monitor_node_global(Node, sec_hor, sec_vert):
    """
    Escoge un nodo monitor común para TODOS los casos:
    - cercano al centro de la malla en x
    - fila impar (nodo elevado)
    - lejos de los bordes que tienden a estar más condicionados
    """
    n_row = 2 * sec_vert + 1
    n_col = 2 * sec_hor + 1

    center_col = n_col // 2
    candidate_rows = list(range(1, n_row, 2))   # filas impares -> nodos elevados
    center_row = candidate_rows[len(candidate_rows) // 2]

    idx = center_col * n_row + center_row
    return int(idx)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CARGA
# ═══════════════════════════════════════════════════════════════════════════════

def make_load_right_edge(sec_hor, sec_vert, magnitude=-1.0):
    """
    CASO A: carga vertical uniforme sobre la columna derecha.
    Igual al caso original del repo.
    """
    n_row = 2 * sec_vert + 1
    n_col = 2 * sec_hor + 1
    indp  = np.arange(n_row) + n_row * (n_col - 1)

    fz   = np.full(len(indp), magnitude, dtype=float)
    Load = np.column_stack((
        indp,
        np.zeros_like(indp, dtype=float),
        np.zeros_like(indp, dtype=float),
        fz
    ))
    return Load, indp


def make_load_all_top(sec_hor, sec_vert, total_force=-1.0):
    """
    CASO B: carga vertical uniformemente repartida sobre nodos elevados
    (crestas, filas impares de la grilla).
    No equivale a una presión continua uniforme sobre la superficie completa;
    es una carga nodal concentrada en las crestas.
    """
    n_row = 2 * sec_vert + 1
    n_col = 2 * sec_hor + 1

    loaded = []
    for col in range(n_col):
        for row in range(1, n_row, 2):   # filas impares -> nodos elevados
            idx = col * n_row + row
            loaded.append(idx)
    loaded = np.array(loaded, dtype=int)

    fz_each = total_force / len(loaded)
    fz = np.full(len(loaded), fz_each, dtype=float)

    Load = np.column_stack((
        loaded,
        np.zeros_like(loaded, dtype=float),
        np.zeros_like(loaded, dtype=float),
        fz
    ))
    return Load, loaded


def make_load_weighted(Node, sec_hor, sec_vert, total_force=-1.0):
    """
    CASO C: carga vertical distribuida por área tributaria proyectada (XY),
    aplicada solo en nodos libres en Z.
    """
    n_row   = 2 * sec_vert + 1
    n_col   = 2 * sec_hor + 1
    n_nodes = n_row * n_col

    # Área tributaria nodal en proyección XY
    area_trib = np.zeros(n_nodes)
    for col in range(n_col - 1):
        for row in range(n_row - 1):
            n0 = col * n_row + row
            n1 = n0 + 1
            n2 = (col + 1) * n_row + row
            n3 = n2 + 1

            pts = Node[[n0, n1, n3, n2], :2]

            d1 = pts[2] - pts[0]
            d2 = pts[3] - pts[1]
            area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])

            for n in [n0, n1, n2, n3]:
                area_trib[n] += area / 4.0

    # Excluir nodos en borde izquierdo y todos los restringidos en z
    left_nodes = np.arange(n_row)
    z_restrained = get_z_restrained_nodes(sec_hor, sec_vert)

    mask = np.ones(n_nodes, dtype=bool)
    mask[left_nodes] = False
    mask[z_restrained] = False

    loaded = np.where(mask)[0]

    w = area_trib[loaded]
    w_sum = w.sum()

    if w_sum < 1e-12:
        fz = np.full(len(loaded), total_force / len(loaded), dtype=float)
    else:
        fz = total_force * w / w_sum

    Load = np.column_stack((
        loaded,
        np.zeros_like(loaded, dtype=float),
        np.zeros_like(loaded, dtype=float),
        fz
    ))
    return Load, loaded


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_case(Node_pert, Panel, Supp, Load, label):
    print(f"\n▶ Corriendo caso: {label}")

    BarMater  = lambda Ex: Ogden(Ex, E0)
    RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(
        he, h0, kpi, L0, limlft, limrht
    )

    truss, angles, F = PrepareData(
        Node_pert, Panel, Supp, Load,
        BarMater, RotSpring, Kf, Kb, Abar
    )
    truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

    U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
    U_his  = np.real(U_his)
    LF_his = np.real(LF_his)

    STAT = PostProcess(Data, truss, angles)

    return {
        'truss': truss,
        'angles': angles,
        'F': F,
        'U_his': U_his,
        'LF_his': LF_his,
        'Data': Data,
        'STAT': STAT,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_load_map(Node, Panel, Load_arr, title):
    """
    Vista superior de la malla con magnitud de carga Z por nodo.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for panel in Panel:
        pts = Node[panel, :2]
        pts_c = np.vstack([pts, pts[0]])
        ax.fill(
            pts_c[:, 0], pts_c[:, 1],
            facecolor='#f5f5f5',
            edgecolor='#aaaaaa',
            lw=0.5,
            alpha=0.8
        )

    fz_abs = np.abs(Load_arr[:, 3])
    vmax = fz_abs.max() if fz_abs.max() > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = cm.YlOrRd

    for row in Load_arr:
        idx = int(row[0])
        fz = abs(row[3])
        if fz < 1e-12:
            continue
        ax.scatter(
            Node[idx, 0], Node[idx, 1],
            s=80,
            color=cmap(norm(fz)),
            zorder=5,
            edgecolors='k',
            lw=0.5
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='|Fz| por nodo', shrink=0.85)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()


def plot_curves_comparison(cases_dict, monitor_node):
    """
    Curvas F–δ superpuestas usando el MISMO nodo monitor para todos los casos.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#028090', '#e574bc', '#f4a261', '#2a9d8f']

    dof = monitor_node * 3 + 2  # componente Z

    for (label, res, _), color in zip(cases_dict, colors):
        dsp = res['U_his'][dof, :]
        ax.plot(dsp, res['LF_his'], color=color, lw=2, label=label)

    ax.set_xlabel(f"Desplazamiento Z del nodo {monitor_node} [u]", fontsize=13)
    ax.set_ylabel("Factor de carga λ", fontsize=13)
    ax.set_title("Comparación de esquemas de carga", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_energy_comparison(cases_dict):
    """
    Energía total acumulada vs incremento.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#028090', '#e574bc', '#f4a261', '#2a9d8f']

    for (label, res, _), color in zip(cases_dict, colors):
        pe = res['STAT']['PE']['strain']
        ax.plot(pe, color=color, lw=2, label=label)

    ax.set_xlabel("Incremento", fontsize=13)
    ax.set_ylabel("Energía PE total", fontsize=13)
    ax.set_title("Energía almacenada por esquema de carga", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def print_metrics(cases_dict):
    print("\n" + "═" * 76)
    print("Métricas descriptivas al final del análisis")
    print("─" * 76)
    print(f"{'Caso':<36} {'λ alcanzado':>12} {'|U| máx':>12} {'PE final':>12}")
    print("─" * 76)

    for label, res, _ in cases_dict:
        lf = float(res['LF_his'][-1])
        ud = float(np.max(np.abs(res['U_his'])))
        pe = float(res['STAT']['PE']['strain'][-1])
        print(f"{label:<36} {lf:>12.4f} {ud:>12.4f} {pe:>12.4f}")

    print("═" * 76)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Geometría base con la misma imperfección para todos los casos
    Node, Panel, BDRY = ConfigMiura(sec_hor, sec_vert, theta, a, b, fdang)

    rng = np.random.default_rng(seed)
    Node_pert = Node.copy()
    Node_pert[:, 2] += rng.normal(0.0, perturb_scale, Node.shape[0])

    Supp = make_supports(sec_hor, sec_vert)

    monitor_node = choose_monitor_node_global(Node_pert, sec_hor, sec_vert)
    print(f"\nNodo monitor global para comparar curvas: {monitor_node}")

    # Misma resultante total en los tres casos
    total_force = -(2 * sec_vert + 1) * 1.0

    Load_A, indp_A = make_load_right_edge(sec_hor, sec_vert, magnitude=-1.0)
    Load_B, indp_B = make_load_all_top(sec_hor, sec_vert, total_force=total_force)
    Load_C, indp_C = make_load_weighted(Node_pert, sec_hor, sec_vert, total_force=total_force)

    # Mapas de carga
    plot_load_map(Node_pert, Panel, Load_A,
                  "CASO A — Carga en borde derecho (original)")
    plot_load_map(Node_pert, Panel, Load_B,
                  "CASO B — Carga vertical sobre nodos elevados")
    plot_load_map(Node_pert, Panel, Load_C,
                  "CASO C — Carga vertical por área tributaria proyectada")
    plt.pause(0.1)

    # Simulaciones
    res_A = run_case(Node_pert, Panel, Supp, Load_A, "CASO A — borde derecho")
    res_B = run_case(Node_pert, Panel, Supp, Load_B, "CASO B — nodos elevados")
    res_C = run_case(Node_pert, Panel, Supp, Load_C, "CASO C — área tributaria proyectada")

    cases = [
        ("A — borde derecho", res_A, indp_A),
        ("B — nodos elevados", res_B, indp_B),
        ("C — área tributaria proyectada", res_C, indp_C),
    ]

    # Comparaciones globales
    print_metrics(cases)
    plot_curves_comparison(cases, monitor_node)
    plot_energy_comparison(cases)

    # Pipeline por caso
    for label, res, indp in cases:
        dof_disp = -(monitor_node * 3 + 2)   # mismo nodo monitor, componente Z

        try:
            displacement(res['U_his'], dof_disp, res['LF_his'])
            plt.title(f"Desp. vs. carga — {label}")
        except Exception as e:
            print(f"displacement falló en {label}: {e}")

        try:
            GraphPostProcess(res['U_his'], res['STAT'])
            plt.gcf().suptitle(f"Energía — {label}")
        except Exception as e:
            print(f"GraphPostProcess falló en {label}: {e}")

        try:
            PlotFinalConfig(res['U_his'], res['truss'], res['angles'], res['LF_his'])
            plt.title(f"Config. final — {label}")
        except Exception as e:
            print(f"PlotFinalConfig falló en {label}: {e}")

    plt.show()