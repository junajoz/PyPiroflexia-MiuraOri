import numpy as np


def ConfigMiuraRandomAB(sec_hor, sec_vert, theta_deg, fdang_deg,
                        a_mean, b_mean,
                        a_std=0.02, b_std=0.02,
                        seed=42,
                        a_min=0.5, b_min=0.5):
    """
    Genera una geometría tipo Miura con variación aleatoria de a y b por celda,
    preservando más claramente la irregularidad espacial en la geometría nodal.
    """
    rng = np.random.default_rng(seed)

    theta = np.deg2rad(theta_deg)
    gmma = np.deg2rad(fdang_deg)

    # Parámetros aleatorios por celda
    a_cells = rng.normal(loc=a_mean, scale=a_std, size=(sec_vert, sec_hor))
    b_cells = rng.normal(loc=b_mean, scale=b_std, size=(sec_vert, sec_hor))

    a_cells = np.clip(a_cells, a_min, None)
    b_cells = np.clip(b_cells, b_min, None)

    # Parámetros geométricos locales
    h_cells = a_cells * np.sin(theta) * np.sin(gmma)
    l_cells = a_cells * np.sqrt(1 - (np.sin(gmma) ** 2) * (np.sin(theta) ** 2))
    v_cells = b_cells / np.sqrt(1 + (np.cos(gmma) ** 2) * (np.tan(theta) ** 2))

    # Tamaño de la malla nodal
    n_rows = 2 * sec_vert + 1
    n_cols = 2 * sec_hor + 1

    X = np.zeros((n_rows, n_cols))
    Y = np.zeros((n_rows, n_cols))
    Z = np.zeros((n_rows, n_cols))

    # --------------------------------------------------
    # Construcción de X: dependiente de fila y columna
    # --------------------------------------------------
    for i in range(n_rows):
        row_cell = min(i // 2, sec_vert - 1)
        for j in range(1, n_cols):
            col_cell = min((j - 1) // 2, sec_hor - 1)
            l_local = l_cells[row_cell, col_cell]
            X[i, j] = X[i, j - 1] + l_local / 2.0

    # --------------------------------------------------
    # Construcción de Y: dependiente de fila y columna
    # --------------------------------------------------
    for j in range(n_cols):
        col_cell = min(j // 2, sec_hor - 1)
        for i in range(1, n_rows):
            row_cell = min((i - 1) // 2, sec_vert - 1)
            v_local = v_cells[row_cell, col_cell]
            Y[i, j] = Y[i - 1, j] + v_local

    # --------------------------------------------------
    # Corrimiento alternado Miura en columnas impares
    # usando v local de cada columna
    # --------------------------------------------------
    for j in range(1, n_cols, 2):
        col_cell = min(j // 2, sec_hor - 1)
        for i in range(n_rows):
            row_cell = min(i // 2, sec_vert - 1)
            Y[i, j] += 0.5 * v_cells[row_cell, col_cell]

    # --------------------------------------------------
    # Altura alternada en filas impares
    # usando h local
    # --------------------------------------------------
    for i in range(1, n_rows, 2):
        row_cell = min(i // 2, sec_vert - 1)
        for j in range(n_cols):
            col_cell = min(j // 2, sec_hor - 1)
            Z[i, j] = h_cells[row_cell, col_cell]

    Node = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Paneles cuadriláteros
    Panel = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            n1 = i * n_cols + j
            n2 = n1 + 1
            n3 = n1 + n_cols + 1
            n4 = n1 + n_cols
            Panel.append([n1, n2, n3, n4])
    Panel = np.array(Panel, dtype=int)

    # Bordes
    BDRY = []

    # Superior
    for j in range(n_cols - 1):
        BDRY.append([j, j + 1])

    # Inferior
    base = (n_rows - 1) * n_cols
    for j in range(n_cols - 1):
        BDRY.append([base + j, base + j + 1])

    # Izquierdo
    for i in range(n_rows - 1):
        BDRY.append([i * n_cols, (i + 1) * n_cols])

    # Derecho
    for i in range(n_rows - 1):
        BDRY.append([i * n_cols + (n_cols - 1), (i + 1) * n_cols + (n_cols - 1)])

    BDRY = np.array(BDRY, dtype=int)

    return Node, Panel, BDRY, a_cells, b_cells