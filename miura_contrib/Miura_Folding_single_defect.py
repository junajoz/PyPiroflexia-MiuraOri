import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from ConfigMiuraFromAB import ConfigMiuraFromAB
from defect_fields import build_nominal_fields, apply_single_defect

from Ogden import Ogden
from PrepareData import PrepareData
from PathAnalysis import PathAnalysis
from EnhancedLinear import EnhancedLinear
from PostProcess import PostProcess

from displacement import displacement
from GraphPostProcess import GraphPostProcess
from PlotFinalConfig import PlotFinalConfig
from VisualFold import VisualFold


def run_case(a_cells, b_cells, theta, fdang,
             Kf, Kb, E0, Abar, limlft, limrht,
             sec_hor, sec_vert, blam, MaxIcr):
    Node, Panel, BDRY = ConfigMiuraFromAB(a_cells, b_cells, theta, fdang)

    BarMater = lambda Ex: Ogden(Ex, E0)
    RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(he, h0, kpi, L0, limlft, limrht)

    leftx = np.arange(0, (2 * sec_vert + 1))
    leftz = np.arange(0, (2 * sec_vert + 1) + 1, 2)
    rightz = np.arange(0, (2 * sec_vert + 1) + 1, 2) + (2 * sec_vert + 1) * (2 * sec_hor)

    Supp = np.array([
        [0, 0, 1, 0],
        *zip(leftx, np.ones_like(leftx), np.zeros_like(leftx), np.zeros_like(leftx)),
        *zip(leftz, np.zeros_like(leftz), np.zeros_like(leftz), np.ones_like(leftz)),
        *zip(rightz, np.zeros_like(rightz), np.zeros_like(rightz), np.ones_like(rightz))
    ], dtype=float)

    indp = np.arange(0, (sec_vert * 2 + 1)) + (sec_vert * 2 + 1) * (sec_hor * 2)
    ff = -np.ones(len(indp))
    Load = np.column_stack((indp, ff, np.zeros_like(indp), np.zeros_like(indp)))
    indp = Load[:, 0].astype(int)

    truss, angles, F = PrepareData(Node, Panel, Supp, Load, BarMater, RotSpring, Kf, Kb, Abar)
    truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

    U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
    U_his = np.real(U_his)
    LF_his = np.real(LF_his)

    STAT = PostProcess(Data, truss, angles)

    final_load = float(LF_his[-1]) if len(LF_his) > 0 else np.nan
    max_abs_disp = float(np.max(np.abs(U_his))) if U_his.size > 0 else np.nan

    total_energy = np.nan
    if isinstance(STAT, dict):
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

    return {
        "Node": Node,
        "Panel": Panel,
        "truss": truss,
        "angles": angles,
        "F": F,
        "indp": indp,
        "U_his": U_his,
        "LF_his": LF_his,
        "Data": Data,
        "STAT": STAT,
        "final_load": final_load,
        "max_abs_disp": max_abs_disp,
        "total_energy": total_energy
    }


def logical_cell_to_panel_id(i, j, sec_hor):
    n_cols = 2 * sec_hor + 1
    return (2 * i) * (n_cols - 1) + (2 * j)


def plot_structure_with_defects(Node, Panel, sec_hor, sec_vert, defect_cells=None,
                                title="Estructura con defecto"):
    if defect_cells is None:
        defect_cells = []

    fig, ax = plt.subplots(figsize=(7, 7))

    patches_normal = []
    patches_defect = []

    defect_panel_ids = set()
    for (i, j) in defect_cells:
        if 0 <= i < sec_vert and 0 <= j < sec_hor:
            defect_panel_ids.add(logical_cell_to_panel_id(i, j, sec_hor))

    for pid, panel in enumerate(Panel):
        pts = Node[panel, :2]
        poly = Polygon(pts, closed=True)

        if pid in defect_panel_ids:
            patches_defect.append(poly)
        else:
            patches_normal.append(poly)

    if patches_normal:
        pc_normal = PatchCollection(
            patches_normal,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85
        )
        ax.add_collection(pc_normal)

    if patches_defect:
        pc_defect = PatchCollection(
            patches_defect,
            facecolor="red",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.95
        )
        ax.add_collection(pc_defect)

    ax.scatter(Node[:, 0], Node[:, 1], s=12, color="black")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()


def plot_structure_comparison(Node_nom, Panel_nom, Node_def, Panel_def,
                              sec_hor, sec_vert, defect_cells,
                              title_left="Nominal",
                              title_right="Defectuosa"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, Node_use, Panel_use, title in zip(
        axes,
        [Node_nom, Node_def],
        [Panel_nom, Panel_def],
        [title_left, title_right]
    ):
        patches_normal = []
        patches_defect = []

        defect_panel_ids = set()
        for (i, j) in defect_cells:
            if 0 <= i < sec_vert and 0 <= j < sec_hor:
                defect_panel_ids.add(logical_cell_to_panel_id(i, j, sec_hor))

        for pid, panel in enumerate(Panel_use):
            pts = Node_use[panel, :2]
            poly = Polygon(pts, closed=True)
            if pid in defect_panel_ids:
                patches_defect.append(poly)
            else:
                patches_normal.append(poly)

        if patches_normal:
            pc_normal = PatchCollection(
                patches_normal,
                facecolor="lightgray",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85
            )
            ax.add_collection(pc_normal)

        if patches_defect:
            pc_defect = PatchCollection(
                patches_defect,
                facecolor="red",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.95
            )
            ax.add_collection(pc_defect)

        ax.scatter(Node_use[:, 0], Node_use[:, 1], s=12, color="black")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(True, alpha=0.25)

    plt.tight_layout()


def run_pipeline_plots(case_dict, case_name, do_visual_fold=False):
    """
    Recupera las gráficas del pipeline original para un caso.
    """
    print(f"\nGenerando gráficas del pipeline para: {case_name}")

    U_his = case_dict["U_his"]
    LF_his = case_dict["LF_his"]
    truss = case_dict["truss"]
    angles = case_dict["angles"]
    indp = case_dict["indp"]
    STAT = case_dict["STAT"]

    # Animación opcional
    if do_visual_fold:
        try:
            VisualFold(U_his, truss, angles, LF_his)
        except Exception as e:
            print(f"VisualFold falló en {case_name}: {e}")

    # Curva carga-desplazamiento
    try:
        instdof = -(indp[0] * 3)
        displacement(U_his, instdof, LF_his)
        plt.title(f"Curva carga-desplazamiento - {case_name}")
    except Exception as e:
        print(f"displacement falló en {case_name}: {e}")

    # Gráficas energéticas
    try:
        GraphPostProcess(U_his, STAT)
        plt.gcf().suptitle(f"Postproceso energético - {case_name}")
    except Exception as e:
        print(f"GraphPostProcess falló en {case_name}: {e}")

    # Configuración final
    try:
        PlotFinalConfig(U_his, truss, angles, LF_his)
        plt.title(f"Configuración final - {case_name}")
    except Exception as e:
        print(f"PlotFinalConfig falló en {case_name}: {e}")


if __name__ == "__main__":
    # -----------------------------
    # Parámetros base
    # -----------------------------
    sec_hor = 5
    sec_vert = 3

    theta = 30
    fdang = 15

    a0 = 3.0
    b0 = 2.0

    defect_type = "b"   # 'a', 'b', 'ab'
    severity = 0.20
    defect_cell = (2, 1)

    MaxIcr = 20
    blam = 0.5

    Kf = 1e-1
    Kb = Kf * 1e5
    E0 = 1e6
    Abar = 1e-1

    limlft = 0.1
    limrht = 360 - 0.1

    # Si quieres intentar la animación
    do_visual_fold = False

    # -----------------------------
    # Campo nominal
    # -----------------------------
    a_nom, b_nom = build_nominal_fields(sec_hor, sec_vert, a0, b0)

    # -----------------------------
    # Caso nominal
    # -----------------------------
    nominal = run_case(
        a_nom, b_nom, theta, fdang,
        Kf, Kb, E0, Abar, limlft, limrht,
        sec_hor, sec_vert, blam, MaxIcr
    )

    # -----------------------------
    # Caso con defecto local
    # -----------------------------
    a_def, b_def = apply_single_defect(
        a_nom, b_nom,
        defect_cell[0], defect_cell[1],
        defect_type=defect_type,
        severity=severity
    )

    defect = run_case(
        a_def, b_def, theta, fdang,
        Kf, Kb, E0, Abar, limlft, limrht,
        sec_hor, sec_vert, blam, MaxIcr
    )

    # -----------------------------
    # Comparación de métricas
    # -----------------------------
    delta_final_load = defect["final_load"] - nominal["final_load"]
    delta_max_disp = defect["max_abs_disp"] - nominal["max_abs_disp"]

    if np.isnan(nominal["total_energy"]) or np.isnan(defect["total_energy"]):
        delta_energy = np.nan
    else:
        delta_energy = defect["total_energy"] - nominal["total_energy"]

    print("\n================ RESULTADOS ================\n")
    print(f"Celda defectuosa: {defect_cell}")
    print(f"Tipo de defecto : {defect_type}")
    print(f"Severidad       : {severity:.2f}")

    print("\n--- Caso nominal ---")
    print(f"final_load   = {nominal['final_load']}")
    print(f"max_abs_disp = {nominal['max_abs_disp']}")
    print(f"total_energy = {nominal['total_energy']}")

    print("\n--- Caso defectuoso ---")
    print(f"final_load   = {defect['final_load']}")
    print(f"max_abs_disp = {defect['max_abs_disp']}")
    print(f"total_energy = {defect['total_energy']}")

    print("\n--- Diferencias respecto al nominal ---")
    print(f"delta_final_load = {delta_final_load}")
    print(f"delta_max_disp   = {delta_max_disp}")
    print(f"delta_energy     = {delta_energy}")

    # -----------------------------
    # Visualización local del defecto
    # -----------------------------
    plot_structure_with_defects(
        nominal["Node"], nominal["Panel"],
        sec_hor, sec_vert,
        defect_cells=[defect_cell],
        title=f"Estructura nominal con defecto marcado en {defect_cell}"
    )

    plot_structure_comparison(
        nominal["Node"], nominal["Panel"],
        defect["Node"], defect["Panel"],
        sec_hor, sec_vert,
        defect_cells=[defect_cell],
        title_left="Geometría nominal",
        title_right=f"Geometría con defecto en {defect_cell}"
    )

    # -----------------------------
    # Gráficas del pipeline original
    # -----------------------------
    run_pipeline_plots(nominal, "Caso nominal", do_visual_fold=do_visual_fold)
    run_pipeline_plots(defect, f"Caso defectuoso {defect_cell}", do_visual_fold=do_visual_fold)

    plt.show()