# This Python script is a simulation and analysis program for Miura folding, a type of origami folding
# pattern

 
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
# Ref: K. Liu, G. H. Paulino (2017). 'Nonlinear mechanics of non-rigid origami - An efficient computational approach.'
#      Proceedings of the Royal Society A.
#      K. Liu, G. H. Paulino (2016). 'MERLIN: A MATLAB implementation to capture highly nonlinear behavior of non-rigid origami.'
#      Proceedings of IASS Annual Symposium 2016.

# MIURA FOLDING

# Define geometry and material parameters
sec_hor = 5  # Number of unit cells in horizontal direction
sec_vert = 5  # Number of unit cells in vertical direction
# Geometry of the Miura: a, b are the edge lengths of each parallelogram
# panel; fdang controls the folding angle; theta is the panel angle
theta = 60
a = 2
b = 2
fdang = 15

# Maximum increment number & initial load factor
MaxIcr = 60
blam = 0.5

# Material-related parameters: Kf-folding stiffness; Kb-bending stiffness;
# E0-stretching stiffness; Abar-bar area (assume uniform)
Kf = 1e-1
Kb = Kf * 1e5
E0 = 1e6
Abar = 1e-1

# Left and right limits for the linear range of rotational stiffness
limlft = 0.1
limrht = 360 - 0.1

# Call ConfigMiura to get Node and Panel arrays
Node, Panel, BDRY = ConfigMiura(sec_hor, sec_vert, theta, a, b, fdang)

# -----------------------------
# Random nodal perturbation
# -----------------------------
seed = 42
rng = np.random.default_rng(seed)

perturb_scale = 0.005
Node_pert = Node.copy()

# Z_only perturbation: Add Gaussian noise to the z-coordinate of each node
delta_z = rng.normal(loc=0.0, scale=perturb_scale, size=Node.shape[0])
Node_pert[:, 2] += delta_z

# Define material constitutive and rotational spring functions
BarMater = lambda Ex: Ogden(Ex, E0)  # Define bar material constitutive
RotSpring = lambda he, h0, kpi, L0: EnhancedLinear(he, h0, kpi, L0, limlft, limrht)

# Set up boundary conditions
leftx = np.arange(0, (2 * sec_vert + 1))
leftz = np.arange(0, (2 * sec_vert + 1) + 1, 2)
rightz = np.arange(0, (2 * sec_vert + 1) + 1, 2) + (2 * sec_vert + 1) * (2 * sec_hor)
rightxp = np.arange(1, (2 * sec_vert + 1), 2) + (2 * sec_vert + 1) * (2 * sec_hor)

Supp = np.array([[0, 0, 1, 0],
                 *zip(leftx, np.ones_like(leftx), np.zeros_like(leftx), np.zeros_like(leftx)),
                 *zip(leftz, np.zeros_like(leftz), np.zeros_like(leftz), np.ones_like(leftz)),
                 *zip(rightz, np.zeros_like(rightz), np.zeros_like(rightz), np.ones_like(rightz))])

indp = np.arange(0, (sec_vert * 2 + 1)) + (sec_vert * 2 + 1) * (sec_hor * 2)   # Revisar -1 
ff = -np.ones(len(indp))
Load = np.column_stack((indp, ff, np.zeros_like(indp), np.zeros_like(indp)))
indp = Load[:, 0]

# Perform analysis

truss, angles, F = PrepareData(Node_pert, Panel, Supp, Load, BarMater, RotSpring, Kf, Kb, Abar)
truss['U0'] = np.zeros(3 * truss['Node'].shape[0])

U_his, LF_his, Data = PathAnalysis(truss, angles, F, blam, MaxIcr)
U_his = np.real(U_his)
LF_his = np.real(LF_his)

print(truss)
print(angles)
print(F)

# Visualize simulation
instdof = -(indp[0] * 3 - 2)
interv = 1
endicrm = U_his.shape[1]
# VisualFold(U_his, truss, angles, 'none', 'valley_folding', 0.05, LF_his, instdof, [-np.inf, np.inf, -np.inf, np.inf])
VisualFold(U_his, truss, angles, LF_his)

# Visualize displacement
instdof = -(indp[0]*3)
displacement(U_his, instdof, LF_his)

 
# Plot stored energy vs. pseudo time
# Red line is the total profile.  Between red and cyan is the folding
# energy. Between cyan and magenta is the portion of energy for bending. 
# Below magenta is the stretching energy of bars.
STAT = PostProcess(Data, truss, angles)

GraphPostProcess(U_his, STAT)

PlotFinalConfig(U_his, truss, angles, LF_his)


plt.show()


# print(Node)
# print(Panel)

# from scipy.io import savemat
# variables = {
#     'NODE': Node,
#     'PANEL': Panel,
#     'BDRY': BDRY
# }
# # Save the variables to a .mat file
# savemat('variables.mat', variables)



import os
import json
import numpy as np
from scipy.sparse import coo_matrix
from datetime import datetime

def convert_to_serializable(obj):
    """
    Converts non-serializable objects (e.g., NumPy arrays, coo_matrix) into serializable formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif isinstance(obj, (np.float32, np.float64)):  # Convert NumPy floats to Python floats
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):  # Convert NumPy ints to Python ints
        return int(obj)
    elif isinstance(obj, dict):  # If it's a dictionary, convert its values recursively
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # If it's a list, convert each element recursively
        return [convert_to_serializable(item) for item in obj]
    elif callable(obj):  # If the object is a function, skip it
        return None
    elif isinstance(obj, coo_matrix):  # Handle sparse matrices
        return {
            'data': obj.data.tolist(),
            'row': obj.row.tolist(),
            'col': obj.col.tolist(),
            'shape': obj.shape
        }
    else:
        return str(obj)  # Fallback for other non-serializable objects

def save_variables_history(variables, folder='variable_history'):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Prepare a dictionary to store serializable data
    serializable_data = {}

    # Convert each variable to a JSON-serializable format
    for var_name, var_value in variables.items():
        try:
            serializable_data[var_name] = convert_to_serializable(var_value)
        except Exception as e:
            print(f"Could not serialize {var_name}: {e}")

    # Define the file path
    file_path = os.path.join(folder, f'variables_{timestamp}.json')

    # Save the variables as JSON
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Variables saved to {file_path}")

# Example usage:
variables_to_save = {
    'Node': Node,         # NumPy array
    'Panel': Panel,       # NumPy array or list
    'truss': truss,       # Dictionary containing NumPy arrays and potentially sparse matrices
    'angles': angles,     # Dictionary containing NumPy arrays or lists
    'F': F,               # NumPy array
    'U_his': U_his,       # NumPy array
    'LF_his': LF_his      # NumPy array
}

save_variables_history(variables_to_save)
