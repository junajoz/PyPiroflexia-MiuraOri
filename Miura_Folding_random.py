import numpy as np
from ConfigMiura import ConfigMiura
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

# MERLIN - Ke Liu, Glaucio H. Paulino

# -----------------------------
# 1) Parámetros base
# -----------------------------
sec_hor = 5
sec_vert = 5

theta = 60
a = 2
b = 2
fdang = 15

MaxIcr = 60
blam = 0.5

Kf = 1e-1
Kb = Kf * 1e5
E0 = 1e6
Abar = 1e-1

limlft = 0.1
limrht = 360 - 0.1

# -----------------------------
# 2) Generador aleatorio
# -----------------------------
seed = 42
rng = np.random.default_rng(seed)

# amplitud de perturbación geométrica
# usa valores pequeños para no dañar la geometría
perturb_scale_x = 0.1
perturb_scale_y = 0.1
perturb_scale_z = 0.1

# -----------------------------
# 3) Geometría base
# -----------------------------
Node, Panel, BDRY = ConfigMiura(sec_hor, sec_vert, theta, a, b, fdang)

# -----------------------------
# 4) Perturbación nodal en x, y, z
# -----------------------------
Node_pert = Node.copy()

delta_x = rng.normal(loc=0.0, scale=perturb_scale_x, size=Node.shape[0])
delta_y = rng.normal(loc=0.0, scale=perturb_scale_y, size=Node.shape[0])
delta_z = rng.normal(loc=0.0, scale=perturb_scale_z, size=Node.shape[0])

Node_pert[:, 0] += delta_x
Node_pert[:, 1] += delta_y
Node_pert[:, 2] += delta_z

# -----------------------------
# 5) Constitutivas
# -----------------------------
BarMater = lambda Ex: Ogden(Ex, E0)
RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(he, h0, kpi, L0, limlft, limrht)

# -----------------------------
# 6) Condiciones de borde
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
# 7) Análisis con geometría perturbada
# -----------------------------
truss, angles, F = PrepareData(Node_pert, Panel, Supp, Load, BarMater, RotSpring, Kf, Kb, Abar)
truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
U_his = np.real(U_his)
LF_his = np.real(LF_his)

# -----------------------------
# 8) Visualización
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