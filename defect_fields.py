import numpy as np  
def build_nominal_fields(sec_hor, sec_vert, a0, b0):
    a_cells = np.full((sec_vert, sec_hor), a0, dtype=float)
    b_cells = np.full((sec_vert, sec_hor), b0, dtype=float)
    return a_cells, b_cells

import numpy as np


def build_nominal_fields(sec_hor, sec_vert, a0, b0):
    a_cells = np.full((sec_vert, sec_hor), a0, dtype=float)
    b_cells = np.full((sec_vert, sec_hor), b0, dtype=float)
    return a_cells, b_cells


def apply_single_defect(a_cells, b_cells, i_def, j_def,
                        defect_type="a",
                        severity=0.30,
                        min_value=0.2):
    """
    defect_type:
        'a'  -> degrada solo a_ij
        'b'  -> degrada solo b_ij
        'ab' -> degrada ambos
    severity:
        fracción de reducción. Ej: 0.30 = reduce 30%
    """
    a_new = a_cells.copy()
    b_new = b_cells.copy()

    factor = 1.0 - severity

    if defect_type in ("a", "ab"):
        a_new[i_def, j_def] = max(min_value, a_new[i_def, j_def] * factor)

    if defect_type in ("b", "ab"):
        b_new[i_def, j_def] = max(min_value, b_new[i_def, j_def] * factor)

    return a_new, b_new


def apply_row_defect(a_cells, b_cells, row,
                     defect_type="a",
                     severity=0.30,
                     min_value=0.2):
    a_new = a_cells.copy()
    b_new = b_cells.copy()
    factor = 1.0 - severity

    if defect_type in ("a", "ab"):
        a_new[row, :] = np.maximum(min_value, a_new[row, :] * factor)

    if defect_type in ("b", "ab"):
        b_new[row, :] = np.maximum(min_value, b_new[row, :] * factor)

    return a_new, b_new


def apply_col_defect(a_cells, b_cells, col,
                     defect_type="a",
                     severity=0.30,
                     min_value=0.2):
    a_new = a_cells.copy()
    b_new = b_cells.copy()
    factor = 1.0 - severity

    if defect_type in ("a", "ab"):
        a_new[:, col] = np.maximum(min_value, a_new[:, col] * factor)

    if defect_type in ("b", "ab"):
        b_new[:, col] = np.maximum(min_value, b_new[:, col] * factor)

    return a_new, b_new


def apply_cluster_defect(a_cells, b_cells, center_i, center_j,
                         radius=1,
                         defect_type="a",
                         severity=0.30,
                         min_value=0.2):
    a_new = a_cells.copy()
    b_new = b_cells.copy()
    factor = 1.0 - severity

    nrows, ncols = a_cells.shape

    for i in range(nrows):
        for j in range(ncols):
            if abs(i - center_i) <= radius and abs(j - center_j) <= radius:
                if defect_type in ("a", "ab"):
                    a_new[i, j] = max(min_value, a_new[i, j] * factor)
                if defect_type in ("b", "ab"):
                    b_new[i, j] = max(min_value, b_new[i, j] * factor)

    return a_new, b_new