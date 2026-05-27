import numpy as np
from ConfigMiuraRandomAB import ConfigMiuraRandomAB
from Ogden import Ogden
from PrepareData import PrepareData
from PathAnalysis import PathAnalysis
from EnhancedLinear import EnhancedLinear
from VisualFold import VisualFold
from displacement import displacement
from GraphPostProcess import GraphPostProcess
from PlotFinalConfig import PlotFinalConfig
from PostProcess import PostProcess
from matplotlib import pyplot as plt

def plot_top_view(Node, Panel, title="Vista aérea de la estructura Miura"):
    fig, ax = plt.subplots(figsize=(8, 8))

    for panel in Panel:
        pts = Node[panel, :2]
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], 'k-', linewidth=0.8, alpha=0.8)

    ax.scatter(Node[:, 0], Node[:, 1], s=12)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_ab_maps(a_cells, b_cells):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(a_cells, origin='lower', aspect='auto')
    axes[0].set_title("Mapa de a por celda")
    axes[0].set_xlabel("Celda horizontal")
    axes[0].set_ylabel("Celda vertical")
    fig.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(b_cells, origin='lower', aspect='auto')
    axes[1].set_title("Mapa de b por celda")
    axes[1].set_xlabel("Celda horizontal")
    axes[1].set_ylabel("Celda vertical")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()

# -----------------------------
# 1) Parámetros base
# -----------------------------
sec_hor = 5
sec_vert = 5

theta = 60
fdang = 15

a_mean = 2.0
b_mean = 1.0

a_std = 0.05
b_std = 0.05

seed = 42

MaxIcr = 60
blam = 0.5

Kf = 1e-1
Kb = Kf * 1e5
E0 = 1e6
Abar = 1e-1

limlft = 0.1
limrht = 360 - 0.1

# -----------------------------
# 2) Geometría con a y b aleatorios por celda
# -----------------------------
Node, Panel, BDRY, a_cells, b_cells = ConfigMiuraRandomAB(
    sec_hor=sec_hor,
    sec_vert=sec_vert,
    theta_deg=theta,
    fdang_deg=fdang,
    a_mean=a_mean,
    b_mean=b_mean,
    a_std=a_std,
    b_std=b_std,
    seed=seed
)

print("a por celda:")
print(a_cells)
print("b por celda:")
print(b_cells)

# -----------------------------
# 2.1) Gráficas de inspección geométrica
# -----------------------------
plot_top_view(Node, Panel, title="Vista aérea de la geometría irregular")
plot_ab_maps(a_cells, b_cells)


# -----------------------------
# 3) Constitutivas
# -----------------------------
BarMater = lambda Ex: Ogden(Ex, E0)
RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(he, h0, kpi, L0, limlft, limrht)

# -----------------------------
# 4) Condiciones de borde
# -----------------------------
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

# -----------------------------
# 5) Análisis
# -----------------------------
truss, angles, F = PrepareData(Node, Panel, Supp, Load, BarMater, RotSpring, Kf, Kb, Abar)
truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
U_his = np.real(U_his)
LF_his = np.real(LF_his)

# -----------------------------
# 6) Visualización
# -----------------------------

try:
    VisualFold(U_his, truss, angles, LF_his)
except Exception as e:
    print(f"VisualFold falló: {e}")

instdof = -(indp[0] * 3)
displacement(U_his, instdof, LF_his)

STAT = PostProcess(Data, truss, angles)
GraphPostProcess(U_his, STAT)
PlotFinalConfig(U_his, truss, angles, LF_his)

plt.show()