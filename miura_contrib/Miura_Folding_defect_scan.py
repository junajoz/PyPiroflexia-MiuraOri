import numpy as np
from matplotlib import pyplot as plt

from ConfigMiuraFromAB import ConfigMiuraFromAB
from defect_fields import build_nominal_fields, apply_single_defect

from Ogden import Ogden
from PrepareData import PrepareData
from PathAnalysis import PathAnalysis
from EnhancedLinear import EnhancedLinear
from PostProcess import PostProcess


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
    print("Tipos dentro de STAT:")
    for k, v in STAT.items():
        print("  ", k, type(v))

    # Métricas globales simples
    final_load = float(LF_his[-1]) if len(LF_his) > 0 else np.nan
    max_abs_disp = float(np.max(np.abs(U_his))) if U_his.size > 0 else np.nan

    # Energía total final, si existe en STAT
    total_energy = np.nan
    if isinstance(STAT, dict):
        for key in ["PE", "TotalPE", "Energy", "Etot"]:
            if key not in STAT:
                continue

            value = STAT[key]
            # PostProcess returns STAT['PE'] as a dict e.g. {'strain': ...}
            if isinstance(value, dict):
                if "strain" in value:
                    value = value["strain"]
                else:
                    # Pick the first array-like entry found
                    for candidate in value.values():
                        if isinstance(candidate, (np.ndarray, list, tuple)):
                            value = candidate
                            break

            arr = np.asarray(value).ravel()
            if arr.size > 0:
                try:
                    total_energy = float(arr[-1])
                    break
                except (TypeError, ValueError):
                    # Not a numeric series; keep searching
                    continue

    return {
        "U_his": U_his,
        "LF_his": LF_his,
        "STAT": STAT,
        "final_load": final_load,
        "max_abs_disp": max_abs_disp,
        "total_energy": total_energy
    }


def plot_field(field, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field, origin='lower', aspect='equal')
    ax.set_title(title)
    ax.set_xlabel("Celda horizontal")
    ax.set_ylabel("Celda vertical")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()


if __name__ == "__main__":
    # -----------------------------
    # Parámetros base
    # -----------------------------
    sec_hor = 5
    sec_vert = 5

    theta = 60
    fdang = 15

    a0 = 2.0
    b0 = 2.0

    defect_type = "a"     # 'a', 'b', 'ab'
    severity = 0.30       # 30% de reducción local

    MaxIcr = 60
    blam = 0.5

    Kf = 1e-1
    Kb = Kf * 1e5
    E0 = 1e6
    Abar = 1e-1

    limlft = 0.1
    limrht = 360 - 0.1

    # -----------------------------
    # Campo nominal
    # -----------------------------
    a_nom, b_nom = build_nominal_fields(sec_hor, sec_vert, a0, b0)

    # -----------------------------
    # Corrida nominal
    # -----------------------------
    base = run_case(
        a_nom, b_nom, theta, fdang,
        Kf, Kb, E0, Abar, limlft, limrht,
        sec_hor, sec_vert, blam, MaxIcr
    )

    print("Caso nominal:")
    print("  final_load   =", base["final_load"])
    print("  max_abs_disp =", base["max_abs_disp"])
    print("  total_energy =", base["total_energy"])

    # -----------------------------
    # Escaneo de sensibilidad por celda
    # -----------------------------
    delta_final_load = np.zeros((sec_vert, sec_hor))
    delta_max_disp = np.zeros((sec_vert, sec_hor))
    delta_energy = np.zeros((sec_vert, sec_hor))

    for i in range(sec_vert):
        for j in range(sec_hor):
            a_def, b_def = apply_single_defect(
                a_nom, b_nom, i, j,
                defect_type=defect_type,
                severity=severity
            )

            out = run_case(
                a_def, b_def, theta, fdang,
                Kf, Kb, E0, Abar, limlft, limrht,
                sec_hor, sec_vert, blam, MaxIcr
            )

            delta_final_load[i, j] = out["final_load"] - base["final_load"]
            delta_max_disp[i, j] = out["max_abs_disp"] - base["max_abs_disp"]

            if np.isnan(base["total_energy"]) or np.isnan(out["total_energy"]):
                delta_energy[i, j] = np.nan
            else:
                delta_energy[i, j] = out["total_energy"] - base["total_energy"]

            print(f"Celda ({i},{j}) lista.")

    # -----------------------------
    # Gráficas
    # -----------------------------
    plot_field(delta_final_load, f"Sensibilidad de carga final\nDefecto local tipo {defect_type}")
    plot_field(delta_max_disp, f"Sensibilidad de desplazamiento máximo\nDefecto local tipo {defect_type}")
    plot_field(delta_energy, f"Sensibilidad de energía final\nDefecto local tipo {defect_type}")

    plt.show()