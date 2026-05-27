"""
PathAnalysis_varKpf.py
======================
Idéntico a PathAnalysis.py pero importa los ensambladores corregidos:
  - GlobalK_edu_ver_varKpf  (loop de folds usa kpf[fel], no kpf completo)
  - GlobalK_fast_ver_varKpf (añade assert de shapes)

El único cambio funcional es la importación.
"""

import numpy as np
from scipy.linalg import solve
from nlsmgd  import nlsmgd
from GetData import GetData

from GlobalK_edu_ver_varKpf  import GlobalK_edu_ver_varKpf
from GlobalK_fast_ver_varKpf import GlobalK_fast_ver_varKpf


def PathAnalysis_varKpf(truss, angles, F, b_lambda, MaxIcr):
    """
    Seguimiento de camino no lineal (arc-length / Newton-Raphson).
    Igual que PathAnalysis pero usando los ensambladores con kpf por pliegue.
    """
    tol      = 1e-6
    Maxitera = 50
    Node     = truss['Node']
    AllDofs  = np.arange(0, 3 * Node.shape[0])
    U        = truss['U0']
    Uhis     = np.zeros((3 * Node.shape[0], MaxIcr))

    Data = {
        'Exbar':    np.zeros((truss['Bars'].shape[0],    MaxIcr)),
        'FdAngle':  np.zeros((angles['fold'].shape[0],   MaxIcr)),
        'LFdAngle': np.zeros((angles['fold'].shape[0],   MaxIcr)),
        'BdAngle':  np.zeros((angles['bend'].shape[0],   MaxIcr)),
        'LBdAngle': np.zeros((angles['bend'].shape[0],   MaxIcr)),
    }

    FreeDofs = np.setdiff1d(AllDofs, truss['FixedDofs'])
    lmd      = 0
    icrm     = 0
    MUL      = np.column_stack((U, U))
    load_his = np.zeros(MaxIcr)

    while icrm < MaxIcr:
        icrm  += 1
        itera  = 0
        err    = 1
        print(f'icrm = {icrm}, lambda = {lmd:6.4f}')

        while err > tol and itera < Maxitera:
            itera += 1

            # ── Ensambladores corregidos ──────────────────────────────────────
            IF2, K2 = GlobalK_edu_ver_varKpf(U, Node, truss, angles)
            IF,  K  = GlobalK_fast_ver_varKpf(U, Node, truss, angles)

            R = lmd * F - IF.T
            R[np.isnan(R)] = 0
            MRS = np.column_stack((F, R.T))

            MUL[FreeDofs, :] = solve(
                K[np.ix_(FreeDofs, FreeDofs)].toarray(),
                MRS[FreeDofs, :]
            )
            dUp = MUL[:, 0]
            dUr = MUL[:, 1]

            if itera == 1:
                dUr = np.zeros_like(dUr)

            dlmd  = nlsmgd(icrm, itera, dUp, dUr, b_lambda)
            dUt   = dlmd * dUp + dUr
            U     = U + dUt
            err   = np.linalg.norm(dUt[FreeDofs])
            lmd   = lmd + dlmd

            print(f'    itera = {itera}, err = {err:6.4f}, dlambda = {dlmd:6.4f}')

            if err > 1e8:
                print('Divergence!')
                break

        # ── Control del radio de arco ─────────────────────────────────────────
        if itera > 15:
            b_lambda = b_lambda / 2
            print('Reduce constraint radius!')
            icrm -= 1
            U    = Uhis[:, max(icrm, 1) - 1]
            lmd  = load_his[max(icrm, 1) - 1]

        elif itera < 3:
            print('Increase constraint radius!')
            b_lambda = b_lambda * 1.5
            Uhis[:, icrm - 1] = U
            load_his[icrm - 1] = lmd
            Exbari, FdAnglei, BdAnglei, LFdAnglei, LBdAnglei = \
                GetData(U, Node, truss, angles)
            Data['Exbar']   [:, icrm - 1] = Exbari.T
            Data['FdAngle'] [:, icrm - 1] = FdAnglei.T
            Data['LFdAngle'][:, icrm - 1] = LFdAnglei.T
            Data['BdAngle'] [:, icrm - 1] = BdAnglei.T
            Data['LBdAngle'][:, icrm - 1] = LBdAnglei.T

        else:
            Uhis[:, icrm - 1] = U
            load_his[icrm - 1] = lmd
            Exbari, FdAnglei, BdAnglei, LFdAnglei, LBdAnglei = \
                GetData(U, Node, truss, angles)
            Data['Exbar']   [:, icrm - 1] = Exbari.T
            Data['FdAngle'] [:, icrm - 1] = FdAnglei.T
            Data['LFdAngle'][:, icrm - 1] = LFdAnglei.T
            Data['BdAngle'] [:, icrm - 1] = BdAnglei.T
            Data['LBdAngle'][:, icrm - 1] = LBdAnglei.T

    icrm += 1
    Uhis     = Uhis    [:, :icrm]
    load_his = load_his[:icrm]
    for key in ['Exbar', 'FdAngle', 'LFdAngle', 'BdAngle', 'LBdAngle']:
        Data[key] = Data[key][:, :icrm]

    return Uhis, load_his, Data
