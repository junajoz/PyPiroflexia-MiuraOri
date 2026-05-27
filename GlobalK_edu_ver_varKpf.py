"""
GlobalK_edu_ver_varKpf.py
=========================
Versión corregida de GlobalK_edu_ver que consume angles['kpf'] como array
1-D de longitud n_folds (un valor por pliegue).

BUG ORIGINAL (línea 101 de GlobalK_edu_ver.py):
    FoldKe(Nodenw, fold, angles['kpf'], ...)
                         ^^^^^^^^^^^^^^
    Se pasaba el array COMPLETO de kpf a cada llamada a FoldKe en lugar
    del escalar correspondiente al pliegue `fel`. FoldKe recibe un escalar
    o un array de 1 elemento; al recibir el array completo tomaba siempre
    el primer valor → todos los pliegues usaban kpf[0], haciendo invisibles
    las diferencias entre casos.

CORRECCIÓN:
    angles['kpf'][fel]   ← escalar del pliegue actual
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from BarKe import BarKe
from FoldKe import FoldKe


def GlobalK_edu_ver_varKpf(Ui, Node, truss, angles):
    """
    Igual que GlobalK_edu_ver pero usando angles['kpf'][fel] (escalar)
    en cada iteración del loop de pliegues.
    """
    Nn = Node.shape[0]
    IFb = np.zeros((3 * Nn, 1))
    IFp = IFb.copy()
    indi   = np.zeros(36 * truss['Bars'].shape[0])
    indj   = indi.copy()
    kentry = indi.copy()

    Nodenw = Node.copy()
    Nodenw[:, 0] += Ui[::3]
    Nodenw[:, 1] += Ui[1::3]
    Nodenw[:, 2] += Ui[2::3]

    Kb  = csr_matrix((3 * Nn, 3 * Nn))
    Kbd = Kb.copy()
    Kfd = Kb.copy()

    B_dense = truss['B'].todense()

    # ── Barras ────────────────────────────────────────────────────────────────
    for bel in range(truss['Bars'].shape[0]):
        eDof = np.array([
            np.arange(0, 3) + truss['Bars'][bel, 0] * 3,
            np.arange(0, 3) + truss['Bars'][bel, 1] * 3
        ], dtype='int').ravel()
        _, Rbe, Kbe = BarKe(
            Ui[eDof],
            np.array(B_dense[bel, eDof]).flatten(),
            truss['L'][bel],
            truss['CM'],
            truss['A'][bel]
        )
        IFb[eDof, :] = (IFb[eDof, :].T + Rbe).T
        I = np.repeat(eDof, 6).reshape(6, 6).T
        J = I.copy().T
        indi  [36 * bel:36 * (bel + 1)] = I.ravel()
        indj  [36 * bel:36 * (bel + 1)] = J.ravel()
        kentry[36 * bel:36 * (bel + 1)] = Kbe.ravel()

    Kb = csr_matrix((kentry, (indi, indj)),
                    shape=(3 * Nn, 3 * Nn), dtype=np.float64)

    # ── Dobleces (bend) ───────────────────────────────────────────────────────
    indi   = np.zeros(144 * angles['bend'].shape[0])
    indj   = indi.copy()
    kentry = indi.copy()
    Lbend  = truss['L'][:angles['bend'].shape[0]]

    for d_el in range(angles['bend'].shape[0]):
        eDof = np.array([
            3 * angles['bend'][d_el, :],
            3 * angles['bend'][d_el, :] + 1,
            3 * angles['bend'][d_el, :] + 2
        ]).T.ravel()
        bend = angles['bend'][d_el, :]

        # kpb es siempre escalar (sin cambio)
        _, Rpe, Kpe = FoldKe(
            Nodenw, bend,
            angles['kpb'],                      # escalar o array → igual que antes
            np.array([angles['pb0'][d_el]]),
            Lbend[d_el],
            angles['CM']
        )
        IFp[eDof, :] = (IFp[eDof, :].T + Rpe).T
        I = np.repeat(eDof, 12).reshape(12, 12).T
        J = I.copy().T
        indi  [144 * d_el:144 * (d_el + 1)] = I.ravel()
        indj  [144 * d_el:144 * (d_el + 1)] = J.ravel()
        kentry[144 * d_el:144 * (d_el + 1)] = Kpe.ravel()

    Kbd = csr_matrix((kentry, (indi, indj)),
                     shape=(3 * Nn, 3 * Nn), dtype=np.float64)

    # ── Pliegues (fold) — CORRECCIÓN AQUÍ ────────────────────────────────────
    indi   = np.zeros(144 * angles['fold'].shape[0])
    indj   = indi.copy()
    kentry = indi.copy()
    Lfold  = truss['L'][
        len(angles['bend']):
        len(angles['bend']) + len(angles['fold'])
    ]

    for fel in range(angles['fold'].shape[0]):
        eDof = np.array([
            3 * angles['fold'][fel, :],
            3 * angles['fold'][fel, :] + 1,
            3 * angles['fold'][fel, :] + 2
        ]).T.ravel()
        fold = angles['fold'][fel, :]

        # ─────────────────────────────────────────────────────────────────────
        # CORRECCIÓN: extraer el escalar del pliegue `fel`
        # ANTES (bug): angles['kpf']          → array completo
        # AHORA:       angles['kpf'][fel]      → escalar correcto
        # ─────────────────────────────────────────────────────────────────────
        kpf_el = float(angles['kpf'][fel])

        _, Rpe, Kpe = FoldKe(
            Nodenw, fold,
            kpf_el,                             # ← escalar por pliegue
            np.array([angles['pf0'][fel]]),
            Lfold[fel],
            angles['CM']
        )
        IFp[eDof, :] = (IFp[eDof, :].T + Rpe).T
        I = np.repeat(eDof, 12).reshape(12, 12).T
        J = I.copy().T
        indi  [144 * fel:144 * (fel + 1)] = I.ravel()
        indj  [144 * fel:144 * (fel + 1)] = J.ravel()
        kentry[144 * fel:144 * (fel + 1)] = Kpe.ravel()

    Kfd = csr_matrix((kentry, (indi, indj)), shape=(3 * Nn, 3 * Nn))

    IF = IFb + IFp
    K  = Kb + Kbd + Kfd
    K  = (K + K.T) / 2

    return IF, K
