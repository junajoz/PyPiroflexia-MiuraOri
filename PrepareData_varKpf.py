"""
PrepareData_varKpf.py
=====================
Versión extendida de PrepareData que acepta kpf como escalar uniforme
O como array 1-D de longitud Fold.shape[0] (un valor por pliegue).

Diferencias respecto al original PrepareData.py:
  - kpf puede ser float  → se expande uniformemente (comportamiento original)
  - kpf puede ser ndarray de shape (n_folds,) → se usa directamente
  - Se agrega el campo angles['kpf_map'] con el array final para trazabilidad
  - Todo lo demás es idéntico al original
"""

import numpy as np
from scipy.sparse import csr_matrix
from findbend import findbend
from findfdbd import findfdbd
from dirc3d import dirc3d
from FoldKe import FoldKe


def PrepareData_varKpf(Node, Panel, Supp, Load, BarCM, RotSpring,
                       kpf, kpb, Abar):
    """
    Parámetros
    ----------
    kpf : float  o  ndarray shape (n_folds,)
        Rigidez de resorte de pliegue.
        - float  → uniforme, igual que el PrepareData original.
        - array  → un valor por pliegue (orden igual al que devuelve findfdbd).
    kpb : float
        Rigidez de resorte de doblez (bending). Solo escalar por ahora.
    El resto de parámetros son idénticos a PrepareData.
    """
    # ------------------------------------------------------------------ #
    # Topología
    # ------------------------------------------------------------------ #
    Bend = findbend(Panel, Node)
    Fold, Bdry, Trigl = findfdbd(Panel, Bend)
    Bars = np.vstack([Bend[:, :2], Fold[:, :2], Bdry])
    B, L = dirc3d(Node, Bars.astype(int))

    # ------------------------------------------------------------------ #
    # Condiciones de soporte
    # ------------------------------------------------------------------ #
    if Supp.shape[0] == 0:
        rs = []
    else:
        rs = np.hstack([Supp[:, 0:1]*3,
                        Supp[:, 0:1]*3 + 1,
                        Supp[:, 0:1]*3 + 2]).flatten()
        rs = np.vstack([rs, Supp[:, 1:].flatten()])
        rs = rs[:, rs[1, :] != 0][0]

    # ------------------------------------------------------------------ #
    # Área de barras
    # ------------------------------------------------------------------ #
    if np.isscalar(Abar):
        Abar = np.full(Bars.shape[0], Abar)

    # ------------------------------------------------------------------ #
    # kpf → array de longitud n_folds
    # ------------------------------------------------------------------ #
    n_folds = Fold.shape[0]

    if np.isscalar(kpf):
        kpf_arr = np.full(n_folds, float(kpf))
    else:
        kpf_arr = np.asarray(kpf, dtype=float)
        if kpf_arr.shape != (n_folds,):
            raise ValueError(
                f"kpf tiene shape {kpf_arr.shape} pero se esperaba ({n_folds},). "
                f"Usa build_kpf_field() para construir el array correcto."
            )

    # ------------------------------------------------------------------ #
    # Ángulos iniciales de pliegue pf0
    # Se usa el primer valor de kpf_arr solo como argumento de FoldKe
    # (solo afecta al cálculo del ángulo inicial, no a la rigidez)
    # ------------------------------------------------------------------ #
    pf0 = np.zeros(n_folds)
    for i in range(n_folds):
        pf0[i], _, _ = FoldKe(Node, Fold[i, :], kpf_arr[i], 0)

    # ------------------------------------------------------------------ #
    # Vector de carga
    # ------------------------------------------------------------------ #
    m = Node.shape[0]
    F = np.zeros(3 * m)
    indp = Load[:, 0].astype(int)
    F[3 * indp]     = Load[:, 1]
    F[3 * indp + 1] = Load[:, 2]
    F[3 * indp + 2] = Load[:, 3]

    # ------------------------------------------------------------------ #
    # Diccionarios de salida
    # ------------------------------------------------------------------ #
    truss = {
        'CM':        BarCM,
        'Node':      Node,
        'Bars':      Bars,
        'Trigl':     Trigl,
        'B':         B,
        'L':         L,
        'FixedDofs': np.unique(rs),
        'A':         Abar,
    }

    angles = {
        'CM':      RotSpring,
        'fold':    Fold,
        'bend':    Bend,
        'kpf':     kpf_arr,                          # ← array por pliegue
        'kpb':     np.full(Bend.shape[0], kpb),
        'pf0':     pf0,
        'pb0':     np.full(Bend.shape[0], np.pi),
        'Panel':   Panel,
        'kpf_map': kpf_arr.copy(),                   # copia para trazabilidad
    }

    return truss, angles, F
