import numpy as np


def ConfigMiuraFromAB(a_cells, b_cells, theta_deg, fdang_deg):
    """
    Construye una geometría Miura a partir de campos por celda a_ij y b_ij.
    """
    sec_vert, sec_hor = a_cells.shape

    theta = np.deg2rad(theta_deg)
    gmma = np.deg2rad(fdang_deg)

    h_cells = a_cells * np.sin(theta) * np.sin(gmma)
    l_cells = a_cells * np.sqrt(1 - (np.sin(gmma) ** 2) * (np.sin(theta) ** 2))
    v_cells = b_cells / np.sqrt(1 + (np.cos(gmma) ** 2) * (np.tan(theta) ** 2))

    n_rows = 2 * sec_vert + 1
    n_cols = 2 * sec_hor + 1

    X = np.zeros((n_rows, n_cols))
    Y = np.zeros((n_rows, n_cols))
    Z = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        row_cell = min(i // 2, sec_vert - 1)
        for j in range(1, n_cols):
            col_cell = min((j - 1) // 2, sec_hor - 1)
            l_local = l_cells[row_cell, col_cell]
            X[i, j] = X[i, j - 1] + l_local / 2.0

    for j in range(n_cols):
        col_cell = min(j // 2, sec_hor - 1)
        for i in range(1, n_rows):
            row_cell = min((i - 1) // 2, sec_vert - 1)
            v_local = v_cells[row_cell, col_cell]
            Y[i, j] = Y[i - 1, j] + v_local

    for j in range(1, n_cols, 2):
        col_cell = min(j // 2, sec_hor - 1)
        for i in range(n_rows):
            row_cell = min(i // 2, sec_vert - 1)
            Y[i, j] += 0.5 * v_cells[row_cell, col_cell]

    for i in range(1, n_rows, 2):
        row_cell = min(i // 2, sec_vert - 1)
        for j in range(n_cols):
            col_cell = min(j // 2, sec_hor - 1)
            Z[i, j] = h_cells[row_cell, col_cell]

    Node = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    Panel = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            n1 = i * n_cols + j
            n2 = n1 + 1
            n3 = n1 + n_cols + 1
            n4 = n1 + n_cols
            Panel.append([n1, n2, n3, n4])
    Panel = np.array(Panel, dtype=int)

    BDRY = []

    for j in range(n_cols - 1):
        BDRY.append([j, j + 1])

    base = (n_rows - 1) * n_cols
    for j in range(n_cols - 1):
        BDRY.append([base + j, base + j + 1])

    for i in range(n_rows - 1):
        BDRY.append([i * n_cols, (i + 1) * n_cols])

    for i in range(n_rows - 1):
        BDRY.append([i * n_cols + (n_cols - 1), (i + 1) * n_cols + (n_cols - 1)])

    BDRY = np.array(BDRY, dtype=int)

    return Node, Panel, BDRY