"""
Miura_Folding_cyclic.py
=======================
Numeral 7 — Desplegado / re-plegado cíclico: histéresis energética.

Situación física
----------------
El arc-length de Riks sigue el camino de equilibrio sin importar la
dirección de la carga. Para invertir la dirección basta con:
  1. Terminar la FASE DE CARGA en un incremento definido (snap-through o
     un λ objetivo).
  2. Reiniciar PathAnalysis tomando como estado inicial (U0, λ0) el último
     punto convergido de la fase anterior.
  3. Invertir el vector de referencia F → -F.
     Como el arc-length calcula dlmd libremente, el sistema seguirá la rama
     de descarga/replegado automáticamente.

Cada ciclo completo = FASE_CARGA + FASE_DESCARGA.
Se pueden encadenar N_CICLOS ciclos.

Salidas
-------
  1. Curva F–δ completa del ciclo (lazo de histéresis)
  2. Energía PE total vs. pseudo-tiempo global
  3. Energía disipada por ciclo (área del lazo F–δ, calculada con trapz)
  4. Configuración deformada al final de cada fase
  5. Pipeline completo del último ciclo

Nota sobre convergencia
-----------------------
Al invertir la carga, el primer incremento de la fase de descarga puede
necesitar un radio de arco menor. El parámetro `blam_descarga` permite
ajustarlo independientemente.
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from ConfigMiura      import ConfigMiura
from PrepareData      import PrepareData
from PathAnalysis     import PathAnalysis
from PostProcess      import PostProcess
from Ogden            import Ogden
from EnhancedLinear   import EnhancedLinear
from displacement     import displacement
from GraphPostProcess import GraphPostProcess
from PlotFinalConfig  import PlotFinalConfig


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

MaxIcr_carga     = 60
MaxIcr_descarga  = 80
MaxIcr_recarga   = 60

blam_carga    = 0.5
blam_descarga = 1e-3
blam_recarga  = 1e-3

perturb_scale = 0.005
seed          = 42

lambda_objetivo = 0.70
lambda_tol      = 5e-3
dlmd_max        = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# RESTART ARC-LENGTH
# ═══════════════════════════════════════════════════════════════════════════════
def PathAnalysis_restart(truss, angles, F, b_lambda, MaxIcr, U0, lmd0,
                         predictor_sign=None, dlmd_max=10.0,
                         b_lambda_min=1e-12):
    from scipy.linalg import solve
    from GlobalK_edu_ver import GlobalK_edu_ver
    from GlobalK_fast_ver import GlobalK_fast_ver
    from nlsmgd import nlsmgd
    from GetData import GetData

    tol = 1e-6
    Maxitera = 50

    Node = truss['Node']
    ndofs = 3 * Node.shape[0]
    AllDofs = np.arange(ndofs)
    FreeDofs = np.setdiff1d(AllDofs, truss['FixedDofs'])

    U = U0.copy()
    lmd = float(lmd0)

    U_last = U0.copy()
    lmd_last = float(lmd0)

    Uhis = np.zeros((ndofs, MaxIcr))
    load_his = np.zeros(MaxIcr)

    Data = {
        'Exbar':    np.zeros((truss['Bars'].shape[0],  MaxIcr)),
        'FdAngle':  np.zeros((angles['fold'].shape[0], MaxIcr)),
        'LFdAngle': np.zeros((angles['fold'].shape[0], MaxIcr)),
        'BdAngle':  np.zeros((angles['bend'].shape[0], MaxIcr)),
        'LBdAngle': np.zeros((angles['bend'].shape[0], MaxIcr)),
    }

    icrm = 0

    while icrm < MaxIcr:
        U_base = U_last.copy()
        lmd_base = lmd_last

        icrm += 1
        itera = 0
        err = 1.0
        converged = False

        MUL = np.zeros((ndofs, 2))

        print(f'  icrm={icrm}, λ={lmd:10.6f}, b_lambda={b_lambda:.4e}')

        while err > tol and itera < Maxitera:
            itera += 1

            IF2, K2 = GlobalK_edu_ver(U, Node, truss, angles)
            IF, K = GlobalK_fast_ver(U, Node, truss, angles)

            R = lmd * F - IF.T
            R[np.isnan(R)] = 0.0
            MRS = np.column_stack((F, R.T))

            try:
                MUL[FreeDofs, :] = solve(
                    K[np.ix_(FreeDofs, FreeDofs)].toarray(),
                    MRS[FreeDofs, :]
                )
            except Exception as e:
                print(f'    solve failed: {e}')
                err = 1e16
                break

            dUp = MUL[:, 0]
            dUr = MUL[:, 1]

            if itera == 1:
                dUr = np.zeros_like(dUr)

            dlmd = nlsmgd(icrm, itera, dUp, dUr, b_lambda)

            if icrm == 1 and itera == 1 and predictor_sign is not None:
                dlmd = predictor_sign * abs(dlmd)

            if (not np.isfinite(dlmd)) or abs(dlmd) > dlmd_max:
                print(f'    dlmd unstable ({dlmd:.4e}) -> reject increment')
                err = 1e16
                break

            dUt = dlmd * dUp + dUr

            U_trial = U + dUt
            lmd_trial = lmd + dlmd

            if (not np.all(np.isfinite(U_trial))) or (not np.isfinite(lmd_trial)):
                print('    non-finite trial state -> reject increment')
                err = 1e16
                break

            U = U_trial
            lmd = lmd_trial
            err = np.linalg.norm(dUt[FreeDofs])

            print(f'    itera={itera}, err={err:.4e}, Δλ={dlmd:.6f}, λ={lmd:.6f}')

            if err > 1e8:
                print('    Divergence!')
                err = 1e16
                break

        if err <= tol:
            converged = True

        if not converged:
            b_lambda = b_lambda / 2.0
            print(f'  Reduce arc radius -> b_lambda = {b_lambda:.4e}')

            U = U_base.copy()
            lmd = lmd_base
            icrm -= 1

            if b_lambda < b_lambda_min:
                raise RuntimeError(
                    f"Arc-length radius became too small ({b_lambda:.3e})."
                )
        else:
            from GetData import GetData
            Uhis[:, icrm - 1] = U
            load_his[icrm - 1] = lmd

            Exbari, FdAnglei, BdAnglei, LFdAnglei, LBdAnglei = GetData(
                U, Node, truss, angles
            )

            Data['Exbar'][:, icrm - 1] = Exbari.T
            Data['FdAngle'][:, icrm - 1] = FdAnglei.T
            Data['LFdAngle'][:, icrm - 1] = LFdAnglei.T
            Data['BdAngle'][:, icrm - 1] = BdAnglei.T
            Data['LBdAngle'][:, icrm - 1] = LBdAnglei.T

            U_last = U.copy()
            lmd_last = float(lmd)

            if itera < 3:
                b_lambda = b_lambda * 1.5
                print(f'  Increase arc radius -> b_lambda = {b_lambda:.4e}')

    Uhis = Uhis[:, :icrm]
    load_his = load_his[:icrm]
    for key in ['Exbar', 'FdAngle', 'LFdAngle', 'BdAngle', 'LBdAngle']:
        Data[key] = Data[key][:, :icrm]

    return Uhis, load_his, Data, U, lmd


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════
def concat_data(D1, D2):
    out = {}
    for key in D1:
        out[key] = np.concatenate([D1[key], D2[key]], axis=-1)
    return out


def clip_until_lambda_target(Uhis, LF, Data, target, mode='above'):
    """
    Recorta el historial hasta el primer punto donde λ alcanza el target.
    mode='above'  -> primer índice con λ >= target
    mode='below'  -> primer índice con λ <= target
    """
    idx = None
    if mode == 'above':
        candidates = np.where(LF >= target)[0]
    else:
        candidates = np.where(LF <= target)[0]

    if len(candidates) == 0:
        return Uhis, LF, Data

    idx = candidates[0] + 1  # incluir ese punto
    Uhis2 = Uhis[:, :idx]
    LF2 = LF[:idx]
    Data2 = {k: v[:, :idx] for k, v in Data.items()}
    return Uhis2, LF2, Data2


def hysteresis_area(dsp, lf):
    return abs(np.trapezoid(lf, dsp))


def plot_hysteresis(all_dsp, all_lf, phase_boundaries, dof_label):
    fig, ax = plt.subplots(figsize=(9, 6))

    # desplazamiento relativo
    all_dsp = all_dsp - all_dsp[0]

    bounds = [0] + list(phase_boundaries) + [len(all_dsp)]
    colors = ['#e63946', '#457b9d', '#f4a261']
    labels = ['Carga', 'Descarga a λ≈0', 'Recarga']

    for k in range(len(bounds)-1):
        i0, i1 = bounds[k], bounds[k+1]
        ax.plot(all_dsp[i0:i1], all_lf[i0:i1],
                lw=2.0,
                marker='o', markersize=3,
                color=colors[k % len(colors)],
                label=labels[k % len(labels)])

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(f"Desplazamiento relativo {dof_label} [u]")
    ax.set_ylabel("Factor de carga λ")
    ax.set_title("Trayectoria carga–descarga–recarga")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # zoom horizontal para que aparezca el lazo
    xmin = np.min(all_dsp)
    xmax = np.max(all_dsp)
    pad = 0.1 * max(abs(xmax - xmin), 1e-8)
    ax.set_xlim(xmin - pad, xmax + pad)

    plt.tight_layout()


def plot_energy_cycles(all_pe, phase_boundaries):
    fig, ax = plt.subplots(figsize=(9, 5))
    t = np.arange(len(all_pe))
    ax.plot(t, all_pe, color='#e63946', lw=2)

    for pb in phase_boundaries:
        ax.axvline(pb, color='gray', lw=1.0, ls='--', alpha=0.7)

    ax.set_xlabel("Incremento global")
    ax.set_ylabel("Energía PE total")
    ax.set_title("Energía almacenada")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Geometría ─────────────────────────────────────────────────────────────
    Node, Panel, BDRY = ConfigMiura(sec_hor, sec_vert, theta, a, b, fdang)
    rng = np.random.default_rng(seed)
    Node_pert = Node.copy()
    Node_pert[:, 2] += rng.normal(0.0, perturb_scale, Node.shape[0])

    # ── Soportes y carga ──────────────────────────────────────────────────────
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
    Load = np.column_stack((indp, ff,
                             np.zeros_like(indp),
                             np.zeros_like(indp)))
    indp = Load[:, 0].astype(int)

    BarMater  = lambda Ex: Ogden(Ex, E0)
    RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(
        he, h0, kpi, L0, limlft, limrht
    )

    truss, angles, F_pos = PrepareData(
        Node_pert, Panel, Supp, Load,
        BarMater, RotSpring, Kf, Kb, Abar
    )
    truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

    dof_monitor = int(indp[0]) * 3 + 0   # X del primer nodo cargado
    dof_label = f"nodo {indp[0]} (X)"
    # ──────────────────────────────────────────────────────────────────────────
    # FASE 1: CARGA HASTA λ_objetivo
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  FASE 1 — CARGA HASTA λ_objetivo")
    print(f"{'═'*55}")

    U_carga, LF_carga, Data_carga, U_end_c, lmd_end_c = PathAnalysis_restart(
        truss, angles, F_pos, blam_carga, MaxIcr_carga,
        truss['U0'], 0.0,
        predictor_sign=+1,
        dlmd_max=dlmd_max
    )

    U_carga, LF_carga, Data_carga = clip_until_lambda_target(
        U_carga, LF_carga, Data_carga,
        lambda_objetivo, mode='above'
    )

    U_end_c = U_carga[:, -1].copy()
    lmd_end_c = float(LF_carga[-1])

    print(f"\nλ final de carga recortada = {lmd_end_c:.6f}")

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 2: DESCARGA CONTROLADA HASTA λ≈0
    # Manteniendo F_pos, pero forzando predictor negativo.
    # NO cambiamos a -F_pos aquí.
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  FASE 2 — DESCARGA CONTROLADA HASTA λ≈0")
    print(f"{'═'*55}")

    U_desc, LF_desc, Data_desc, U_end_d, lmd_end_d = PathAnalysis_restart(
        truss, angles, F_pos, blam_descarga, MaxIcr_descarga,
        U_end_c, lmd_end_c,
        predictor_sign=-1,
        dlmd_max=dlmd_max
    )

    U_desc, LF_desc, Data_desc = clip_until_lambda_target(
        U_desc, LF_desc, Data_desc,
        lambda_tol, mode='below'
    )

    U_end_d = U_desc[:, -1].copy()
    lmd_end_d = float(LF_desc[-1])

    print(f"\nλ final de descarga recortada = {lmd_end_d:.6f}")

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 3: RECARGA HASTA λ_objetivo
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  FASE 3 — RECARGA HASTA λ_objetivo")
    print(f"{'═'*55}")

    U_rec, LF_rec, Data_rec, U_end_r, lmd_end_r = PathAnalysis_restart(
        truss, angles, F_pos, blam_recarga, MaxIcr_recarga,
        U_end_d, lmd_end_d,
        predictor_sign=+1,
        dlmd_max=dlmd_max
    )

    U_rec, LF_rec, Data_rec = clip_until_lambda_target(
        U_rec, LF_rec, Data_rec,
        lambda_objetivo, mode='above'
    )

    # ──────────────────────────────────────────────────────────────────────────
    # CONCATENACIÓN
    # ──────────────────────────────────────────────────────────────────────────
    all_U_his = np.concatenate([U_carga, U_desc, U_rec], axis=1)
    all_LF_his = np.concatenate([LF_carga, LF_desc, LF_rec])
    all_Data = concat_data(concat_data(Data_carga, Data_desc), Data_rec)

    phase_boundaries = [
        U_carga.shape[1],
        U_carga.shape[1] + U_desc.shape[1]
    ]

    loaded_dofs_x = indp * 3 + 0
    dof_label = "promedio borde cargado (X)"

    # ──────────────────────────────────────────────────────────────────────────
    # POST
    # ──────────────────────────────────────────────────────────────────────────
    STAT_global = PostProcess(all_Data, truss, angles)
    PE_total = STAT_global['PE']['strain']

    all_dsp = np.mean(all_U_his[loaded_dofs_x, :], axis=0)

    # Área del lazo principal: carga + descarga
    dsp_c = np.mean(U_carga[loaded_dofs_x, :], axis=0)
    dsp_d = np.mean(U_desc[loaded_dofs_x, :], axis=0)
    dsp_loop = np.concatenate([dsp_c, dsp_d])
    lf_loop = np.concatenate([LF_carga, LF_desc])
    area = hysteresis_area(dsp_loop, lf_loop)

    print("\n" + "═"*48)
    print(f"Área del lazo carga–descarga: {area:.6f}")
    print("═"*48)

    # ──────────────────────────────────────────────────────────────────────────
    # GRÁFICAS
    # ──────────────────────────────────────────────────────────────────────────
    plot_hysteresis(all_dsp, all_LF_his, phase_boundaries, dof_label)
    plot_energy_cycles(PE_total, phase_boundaries)

    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        dsp_c_plot = dsp_c - dsp_c[0]
        ax.plot(dsp_c_plot, LF_carga, '-o', ms=3, lw=2)
        ax.set_xlabel("Desplazamiento relativo medio borde cargado X [u]")
        ax.set_ylabel("Factor de carga λ")
        ax.set_title("Despl. medio vs. carga — fase de carga")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    except Exception as e:
        print(f"plot carga: {e}")

    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        dsp_d_plot = dsp_d - dsp_d[0]
        ax.plot(dsp_d_plot, LF_desc, '-o', ms=3, lw=2)
        ax.set_xlabel("Desplazamiento relativo medio borde cargado X [u]")
        ax.set_ylabel("Factor de carga λ")
        ax.set_title("Despl. medio vs. carga — fase de descarga")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    except Exception as e:
        print(f"plot descarga: {e}")

    try:
        GraphPostProcess(all_U_his, STAT_global)
        plt.gcf().suptitle("Energía del ciclo completo")
    except Exception as e:
        print(f"GraphPostProcess: {e}")

    try:
        PlotFinalConfig(all_U_his, truss, angles, all_LF_his)
        plt.title("Configuración final")
    except Exception as e:
        print(f"PlotFinalConfig: {e}")

    plt.show()