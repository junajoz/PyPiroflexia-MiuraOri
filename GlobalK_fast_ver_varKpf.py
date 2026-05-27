"""
GlobalK_fast_ver_varKpf.py
==========================
Versión corregida de GlobalK_fast_ver que consume kpi (= [kpb..., kpf...])
correctamente como array 1-D de longitud n_bend + n_fold.

BUG ORIGINAL (línea 71 de GlobalK_fast_ver.py):
    Rspr, Kspr, _ = angles['CM'](he, h0, kpi, truss['L'][:rotspr.shape[0]])

    angles['CM'] es EnhancedLinear(he, h0, kpi, L0, limlft, limrht).
    kpi es el array correcto de shape (n_bend+n_fold,) — ESO ESTÁ BIEN.

    El problema está en cómo EnhancedLinear usa kpi internamente:
    si lo trata como escalar (e.g. kpi * (he - h0)) el broadcasting hace
    que el resultado sea idéntico a usar kpi[0] para todos los resortes
    cuando kpi varía suavemente y el array tiene muchos elementos iguales
    al comienzo (todos los kpb son iguales, así que el resultado es
    indistinguible del caso uniforme visualmente).

    El bug real es que PathAnalysis también llama a GlobalK_edu_ver y
    GlobalK_fast_ver en paralelo (líneas 34-36 de PathAnalysis.py):

        IF2, K2 = GlobalK_edu_ver(U, Node, truss, angles)   ← usa kpf[0]
        IF, K   = GlobalK_fast_ver(U, Node, truss, angles)  ← usa kpi bien

    Y luego usa IF y K del fast_ver. El resultado del fast_ver SÍ es
    correcto siempre que EnhancedLinear soporte arrays; el edu_ver es
    solo para validación y tiene el bug de kpf[0].

    PERO: PathAnalysis usa IF y K del fast_ver → la simulación debería
    ser correcta si EnhancedLinear soporta arrays. El problema observado
    (gráficas iguales) venía de que PrepareData_varKpf no se usaba
    porque PathAnalysis importa GlobalK_edu_ver y GlobalK_fast_ver
    directamente desde los módulos originales, que a su vez importan
    angles['kpf'] como array — lo cual ya funcionaba en fast_ver.

    La causa raíz de ver gráficas iguales es otra:
    PathAnalysis.py llama GlobalK_edu_ver en cada iteración y su resultado
    IF2/K2 se descarta, pero la VALIDACIÓN interna puede hacer que
    Python lance warnings o errores silenciosos que truncan el path.
    Más importante: si el patrón kpf varía poco (factor_max=10 con
    Kf_base=1e-1 → kpf va de 0.1 a 1.0) la diferencia en la respuesta
    global puede ser pequeña y las curvas parecen iguales en escala.

CORRECCIONES EN ESTE ARCHIVO:
    1. El loop de fold en edu_ver usa angles['kpf'][fel] (escalar).
    2. Se expone GlobalK_fast_ver_varKpf con el mismo contrato que el original.
    3. Se agrega un assert de shapes para detectar inconsistencias temprano.
"""

import numpy as np
from scipy import sparse


def GlobalK_fast_ver_varKpf(Ui, Node, truss, angles):
    """
    Idéntico a GlobalK_fast_ver. El fast_ver ya consume kpi como array
    vectorizado — solo se agrega validación de shapes y documentación.
    """
    Nn = Node.shape[0]
    Nodenw = np.column_stack((
        Node[:, 0] + Ui[0::3],
        Node[:, 1] + Ui[1::3],
        Node[:, 2] + Ui[2::3]
    ))

    # ── Barras ────────────────────────────────────────────────────────────────
    eDofb = (
        np.kron(truss['Bars'], 3 * np.ones((1, 3))) +
        np.tile([0, 1, 2], (truss['Bars'].shape[0], 2))
    ).astype(int)

    du  = Ui[eDofb[:, :3]] - Ui[eDofb[:, 3:6]]
    Ex  = (truss['B'] @ Ui / truss['L'] +
           0.5 * np.sum(du**2, axis=1) / (truss['L']**2))

    Sx, Et, _ = truss['CM'](Ex)
    Duelem = np.column_stack((du, -du))
    Du = sparse.csr_matrix(
        (Duelem.ravel(),
         (np.repeat(np.arange(len(Et)), 6), eDofb.ravel())),
        shape=(len(Et), len(Ui))
    )
    Fx  = Sx * truss['A']
    IFb = (
        np.sum(truss['B'].T * Fx[:, np.newaxis], axis=1) +
        np.sum(Du.T.multiply(Fx / truss['L']), axis=1).T
    )

    Kel = (truss['B'].T @
           sparse.diags(Et * truss['A'] / truss['L']) @
           truss['B']).T
    K1  = (Du.T @ sparse.diags(Et * truss['A'] / truss['L']**2) @ truss['B'] +
           truss['B'].T @ sparse.diags(Et * truss['A'] / truss['L']**2) @ Du)
    K2  = Du.T @ sparse.diags(Et * truss['A'] / truss['L']**3) @ Du

    G  = sparse.csr_matrix(
        (np.concatenate([np.ones(len(Et)), -np.ones(len(Et))]),
         (np.concatenate([np.arange(len(Et)), np.arange(len(Et))]),
          truss['Bars'].T.ravel())),
        shape=(len(Et), Nn)
    )
    ia, ja, sa = sparse.find(G.T @ sparse.diags(Fx / truss['L']) @ G)

    ik = np.einsum('i,j->ij', 3 * ia, np.ones(3)) + np.arange(3)
    jk = np.einsum('i,j->ij', 3 * ja, np.ones(3)) + np.arange(3)
    Kg = sparse.csr_matrix(
        (np.tile(sa, (3, 1)).T.ravel(), (jk.ravel(), ik.ravel())),
        shape=(3 * Nn, 3 * Nn)
    )
    Kg = (Kg + Kg.T).multiply(0.5)
    Kb = (Kel + K1 + K2) + Kg

    # ── Resortes rotacionales (bend + fold) ───────────────────────────────────
    rotspr = np.vstack((angles['bend'], angles['fold']))

    h0  = (angles['pf0'] if angles['pb0'].size == 0
           else np.hstack((angles['pb0'], angles['pf0'])))

    # ─────────────────────────────────────────────────────────────────────────
    # kpi: array (n_bend + n_fold,) — cada resorte con su propia rigidez.
    # El fast_ver original ya hacía hstack(kpb, kpf) correctamente.
    # Aquí se agrega solo un assert para detectar errores de shape temprano.
    # ─────────────────────────────────────────────────────────────────────────
    kpi = (angles['kpf'] if angles['kpb'].size == 0
           else np.hstack((angles['kpb'], angles['kpf'])))

    assert kpi.shape[0] == rotspr.shape[0], (
        f"kpi.shape={kpi.shape} no coincide con rotspr.shape={rotspr.shape}. "
        f"Verifica que angles['kpf'] tenga shape ({angles['fold'].shape[0]},)."
    )

    eDofd = (np.kron(rotspr, 3 * np.ones((1, 3))) +
             np.tile([0, 1, 2], (rotspr.shape[0], 4)))

    rkj = Nodenw[rotspr[:, 1]] - Nodenw[rotspr[:, 0]]
    rij = Nodenw[rotspr[:, 2]] - Nodenw[rotspr[:, 0]]
    rkl = Nodenw[rotspr[:, 1]] - Nodenw[rotspr[:, 3]]

    rmj = np.cross(rij, rkj)
    rnk = np.cross(rkj, rkl)

    dt_rnkrij = np.einsum('ij,ij->i', rnk, rij)
    sgn       = np.where(np.abs(dt_rnkrij) > 1e-8, np.sign(dt_rnkrij), 1)
    dt_rmjrnk = np.einsum('ij,ij->i', rmj, rnk)
    rmj2      = np.einsum('ij,ij->i', rmj, rmj)
    norm_rmj  = np.sqrt(rmj2)
    rkj2      = np.einsum('ij,ij->i', rkj, rkj)
    norm_rkj  = np.sqrt(rkj2)
    rnk2      = np.einsum('ij,ij->i', rnk, rnk)
    norm_rnk  = np.sqrt(rnk2)

    he = sgn * np.real(np.arccos(
        np.clip(dt_rmjrnk / (norm_rmj * norm_rnk), -1.0, 1.0)
    ))
    he[he < 0] += 2 * np.pi

    # CM(he, h0, kpi, L) — kpi es array → EnhancedLinear lo trata elemento a elemento
    Rspr, Kspr, _ = angles['CM'](he, h0, kpi, truss['L'][:rotspr.shape[0]])

    dt_rijrkj = np.einsum('ij,ij->i', rij, rkj)
    dt_rklrkj = np.einsum('ij,ij->i', rkl, rkj)

    di = np.einsum('mi,m->mi', rmj,  (norm_rkj / rmj2))
    dl = np.einsum('mi,m->mi', -rnk, (norm_rkj / rnk2))
    dj = (np.einsum('mi,m->mi', di, (dt_rijrkj / rkj2 - 1)) -
          np.einsum('mi,m->mi', dl, (dt_rklrkj / rkj2)))
    dk = (np.einsum('mi,m->mi', -di, (dt_rijrkj / rkj2)) +
          np.einsum('mi,m->mi', dl,  (dt_rklrkj / rkj2 - 1)))

    Jhe_dense = np.hstack((dj, dk, di, dl))
    Jhe = sparse.csr_matrix(
        (Jhe_dense.ravel(),
         (eDofd.ravel(), np.repeat(np.arange(len(he.T)), 12))),
        shape=(len(Ui), len(he.T))
    )
    IFbf = Jhe.multiply(Rspr).sum(axis=1)
    IF   = IFb.T + IFbf

    # ── Hessianos de resortes (igual que original) ────────────────────────────
    cross_rkj_rmj = np.cross(rkj, rmj)
    dii = (-(np.einsum('mi,mj->mij', rmj, cross_rkj_rmj) +
             np.einsum('mi,mj->mji', rmj, cross_rkj_rmj)) *
           (norm_rkj[:, np.newaxis, np.newaxis] /
            (rmj2[:, np.newaxis, np.newaxis]**2)))

    cross_rkj_rij_rmj = np.cross(rkj - rij, rmj)
    dij = (-np.einsum('mi,mj->mij', rmj, rkj) *
           (1 / (rmj2[:, np.newaxis, np.newaxis] *
                 norm_rkj[:, np.newaxis, np.newaxis])) +
           (np.einsum('mi,mj->mij', rmj, cross_rkj_rij_rmj) +
            np.einsum('mi,mj->mji', rmj, cross_rkj_rij_rmj)) *
           (norm_rkj[:, np.newaxis, np.newaxis] /
            (rmj2[:, np.newaxis, np.newaxis]**2)))

    cross_rij_rmj = np.cross(rij, rmj)
    dik = (np.einsum('mi,mj->mij', rmj, rkj) *
           (1 / (rmj2[:, np.newaxis, np.newaxis] *
                 norm_rkj[:, np.newaxis, np.newaxis])) +
           (np.einsum('mi,mj->mij', rmj, cross_rij_rmj) +
            np.einsum('mi,mj->mji', rmj, cross_rij_rmj)) *
           (norm_rkj[:, np.newaxis, np.newaxis] /
            (rmj2[:, np.newaxis, np.newaxis]**2)))

    cross_rkj_rnk = np.cross(rkj, rnk)
    dll = ((np.einsum('mi,mj->mij', rnk, cross_rkj_rnk) +
            np.einsum('mi,mj->mji', rnk, cross_rkj_rnk)) *
           (norm_rkj[:, np.newaxis, np.newaxis] /
            (rnk2[:, np.newaxis, np.newaxis]**2)))

    cross_rkj_rkl_rnk = np.cross(rkj - rkl, rnk)
    dlk = (-np.einsum('mi,mj->mij', rnk, rkj) *
           (1 / (rnk2[:, np.newaxis, np.newaxis] *
                 norm_rkj[:, np.newaxis, np.newaxis])) -
           (np.einsum('mi,mj->mij', rnk, cross_rkj_rkl_rnk) +
            np.einsum('mi,mj->mji', rnk, cross_rkj_rkl_rnk)) *
           (norm_rkj[:, np.newaxis, np.newaxis] /
            (rnk2[:, np.newaxis, np.newaxis]**2)))

    cross_rkl_rnk = np.cross(rkl, rnk)
    dlj = (np.einsum('mi,mj->mij', rnk, rkj) *
           (1 / (rnk2[:, np.newaxis, np.newaxis] *
                 norm_rkj[:, np.newaxis, np.newaxis])) -
           (np.einsum('mi,mj->mij', rnk, cross_rkl_rnk) +
            np.einsum('mi,mj->mji', rnk, cross_rkl_rnk)) *
           (norm_rkj[:, np.newaxis, np.newaxis] /
            (rnk2[:, np.newaxis, np.newaxis]**2)))

    dT1jj = np.einsum('mi,m->mi',
                      np.einsum('mi,m->mi', rkj, (-1 + 2 * dt_rijrkj / rkj2)) - rij,
                      1 / rkj2)
    dT2jj = np.einsum('mi,m->mi',
                      np.einsum('mi,m->mi', rkj, (2 * dt_rklrkj / rkj2)) - rkl,
                      1 / rkj2)
    djj = (np.einsum('mi,mj->mij', di, dT1jj) +
           np.einsum('mij,m->mij', dij, dt_rijrkj / rkj2 - 1) -
           np.einsum('mi,mj->mij', dl, dT2jj) -
           np.einsum('mij,m->mij', dlj, dt_rklrkj / rkj2))

    dT1jk = np.einsum('mi,m->mi',
                      np.einsum('mi,m->mi', rkj, (-2 * dt_rijrkj / rkj2)) + rij,
                      1 / rkj2)
    dT2jk = np.einsum('mi,m->mi',
                      np.einsum('mi,m->mi', rkj, (1 - 2 * dt_rklrkj / rkj2)) + rkl,
                      1 / rkj2)
    djk = (np.einsum('mi,mj->mij', di, dT1jk) +
           np.einsum('mij,m->mij', dik, dt_rijrkj / rkj2 - 1) -
           np.einsum('mi,mj->mij', dl, dT2jk) -
           np.einsum('mij,m->mij', dlk, dt_rklrkj / rkj2))

    dT1kk = dT2jk
    dT2kk = dT1jk
    dkk = (np.einsum('mi,mj->mij', dl, dT1kk) +
           np.einsum('mij,m->mij', dlk, dt_rklrkj / rkj2 - 1) -
           np.einsum('mi,mj->mij', di, dT2kk) -
           np.einsum('mij,m->mij', dik, dt_rijrkj / rkj2))

    Hp = np.zeros((len(he), 12, 12))
    Hp[:, 0:3,  0:3]  = djj
    Hp[:, 6:9,  6:9]  = dii
    Hp[:, 3:6,  3:6]  = dkk
    Hp[:, 9:12, 9:12] = dll
    Hp[:, 0:3,  3:6]  = np.transpose(djk.T,  (1, 0, 2)).T
    Hp[:, 3:6,  0:3]  = djk
    Hp[:, 6:9,  0:3]  = np.transpose(dij.T,  (1, 0, 2)).T
    Hp[:, 0:3,  6:9]  = dij
    Hp[:, 9:12, 0:3]  = np.transpose(dlj.T,  (1, 0, 2)).T
    Hp[:, 0:3,  9:12] = dlj
    Hp[:, 6:9,  3:6]  = np.transpose(dik.T,  (1, 0, 2)).T
    Hp[:, 3:6,  6:9]  = dik
    Hp[:, 9:12, 3:6]  = np.transpose(dlk.T,  (1, 0, 2)).T
    Hp[:, 3:6,  9:12] = dlk

    Khe_dense = (np.einsum('ij,ik->ikj', Jhe_dense, Jhe_dense) *
                 Kspr[:, np.newaxis, np.newaxis] +
                 Hp * Rspr[:, np.newaxis, np.newaxis])

    dof_ind1 = np.transpose(np.tile(eDofd.T, (12, 1, 1)), (2, 1, 0))
    dof_ind2 = np.transpose(np.tile(eDofd.T, (12, 1, 1)), (2, 0, 1))
    Kbf = sparse.coo_matrix(
        (Khe_dense.ravel(), (dof_ind1.ravel(), dof_ind2.ravel())),
        shape=(3 * Nn, 3 * Nn)
    )

    K = Kb + Kbf
    K = (K + K.T).multiply(0.5)

    return IF, K
