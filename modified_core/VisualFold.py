# import numpy as np
# import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation, writers
# # from PIL import Image
# # from PlotOri import PlotOri
# from plot_ori_panels import plot_panels, save_gif_PIL

# def VisualFold(U_his, truss, angles, LF_his=None):
#     '''The function `VisualFold` generates a 3D animation of a truss structure undergoing deformation and
#     saves it as a GIF file.
    
#     Parameters
#     ----------
#     U_his
#         U_his is a numpy array containing the history of nodal displacements over time in the simulation.
#     Each column represents the nodal displacements at a specific time step.
#     truss
#         The `truss` parameter in the `VisualFold` function seems to represent the truss structure in the
#     simulation. It likely contains information about the nodes and triangles that make up the truss. The
#     function uses this information to visualize the truss structure and its deformation over time.
#     angles
#         The `angles` parameter in the `VisualFold` function seems to represent the angles of the panels in
#     the truss structure. It is used to extract the panel information from the input data. The function
#     then visualizes the truss structure with panels at different angles over time based on the input
#     displacement
#     LF_his
#         LF_his is a NumPy array that represents the load history of the truss structure. It contains the
#     load values applied to the truss at different time steps. If provided, the function will sum the
#     load values along the columns to simplify the visualization.
    
#     '''
#     """
#     Record the simulation to file if needed:
#     recordtype = 'none': do not save the simulation
#     recordtype = 'video': save simulation in MP4 format;
#     recordtype = 'imggif': save simulatrion in GIF format.
#     pausement: pause time between each frame (in seconds).
#                If recordtype = 'none': use a small number such as 0.0001;
#                Otherwise, pausetime = 1/fps;
#     If input data does not include 'load_his', 'instdof', 'axislim', the
#     function does not plot load vs. displacement diagram.
#     axislim: Axis limits (bounding box) for load vs. displacement diagram.
#              format: [xmin,xmax,ymin,ymax].
#     instdof: specify the DOF of interest for displacement measure.
#     """
#     Node = truss['Node']
#     Trigl = truss['Trigl']
#     Panel = angles['Panel']

#     if LF_his is not None and len(LF_his.shape) > 1:
#         LF_his = np.sum(LF_his, axis=1)



 
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     plot_kwargs_wire = {"linewidths": 0.5, "edgecolors": "#858585",
#                    "facecolors": "#d86a96", "alpha": 0.2, "zorder": 4}
#     plot_kwargs = {"linewidths": 1, "edgecolors": "#3c3c3c",
#                    "facecolors": "#d86a96", "zorder": 5}
    
#     files = []
#     Node_history = []

    
#     mins = np.minimum(Node.min(axis=0), Node.min(axis=0))
#     maxs = np.maximum(Node.max(axis=0), Node.max(axis=0))

#     for i in range(U_his.shape[1]):
#         plt.cla()
#         U = U_his[:,i]
#         Nodew = Node.copy()
#         Nodew[:, 0] = Node[:, 0] + U[::3]
#         Nodew[:, 1] = Node[:, 1] + U[1::3]
#         Nodew[:, 2] = Node[:, 2] + U[2::3]

#         # mins = np.minimum(Node.min(axis=0), Nodew.min(axis=0))
#         # maxs = np.maximum(Node.max(axis=0), Nodew.max(axis=0))

#         plot_panels(Node, Panel, ax=ax, **plot_kwargs_wire)
#         plot_panels(Nodew, Panel, ax=ax, **plot_kwargs)    
#         plt.show(block = False)

#         plt.xticks([])
#         plt.yticks([])
#         ax.set_zticks([])
#         ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
#         ax.auto_scale_xyz([mins[0], maxs[0]],
#                         [mins[1], maxs[1]],
#                         [mins[2], maxs[2]])
#         plt.axis("image")
#         file = f"ori_{str(i).zfill(2)}.png"
#         plt.savefig(file)
#         files.append(file)
#         Node_history.append(Nodew)
#         plt.pause(0.001)
    
#     data = {'Node': Node, 'Node_history': Node_history, 'Panel': Panel}
#     np.save('data_completa.npy', data)


#     save_gif_PIL("ori_anim.gif", files, fps=5, loop=0)
#     # [os.remove(file) for file in files]

# import numpy as np
# import pyvista as pv
# from plot_ori_panels import save_gif_PIL
# import os

# def VisualFold(U_his, truss, angles, LF_his=None):
#     '''The function `VisualFold` generates a 3D animation of a truss structure undergoing deformation and
#     saves it as a GIF file.
    
#     Parameters
#     ----------
#     U_his
#         U_his is a numpy array containing the history of nodal displacements over time in the simulation.
#     Each column represents the nodal displacements at a specific time step.
#     truss
#         The `truss` parameter in the `VisualFold` function represents the truss structure in the simulation.
#     angles
#         The `angles` parameter represents the angles of the panels in the truss structure.
#     LF_his
#         LF_his is a NumPy array that represents the load history of the truss structure.
#     '''
    
#     Node = truss['Node']
#     Trigl = truss['Trigl']
#     Panel = angles['Panel']

#     if LF_his is not None and len(LF_his.shape) > 1:
#         LF_his = np.sum(LF_his, axis=1)

#     files = []
#     Node_history = []


#     # Set up PyVista plotter
#     plotter = pv.Plotter(off_screen=False)
#     plotter.add_title("")
#     plotter.show_axes()

#     mins = Node.min(axis=0)
#     maxs = Node.max(axis=0)

#     for i in range(U_his.shape[1]):
#         U = U_his[:, i]
#         Nodew = Node.copy()
#         Nodew[:, 0] += U[::3]
#         Nodew[:, 1] += U[1::3]
#         Nodew[:, 2] += U[2::3]

#         # Clear plotter
#         plotter.clear()

#         # Plot the original structure
#         plot_panels(Node, Panel, plotter, color="#d86a96", opacity=0.3, edge_color="#858585", line_width=0.5)
        
#         # Plot the deformed structure
#         plot_panels(Nodew, Panel, plotter, color="#d86a96", edge_color="#3c3c3c", line_width=1)

#         # Set plot limits
#         plotter.set_scale(xscale=(maxs[0] - mins[0]), yscale=(maxs[1] - mins[1]), zscale=(maxs[2] - mins[2]))
#     plotter.show()

        
#         # # Take a screenshot and save it
#         # file = f"ori_{str(i).zfill(2)}.png"
#         # plotter.screenshot(file)
#         # files.append(file)
#         # Node_history.append(Nodew)

#     # Save GIF
#     save_gif_PIL("ori_anim.gif", files, fps=5, loop=0)

#     # Clean up the files
#     for file in files:
#         os.remove(file)

# def plot_panels(nodes, panels, plotter, color="#d86a96", opacity=1.0, edge_color="#3c3c3c", line_width=1):
#     '''Plot origami panels in 3D as a collection of polygons using PyVista.
    
#     Parameters
#     ----------
#     nodes : ndarray, float
#         Coordinates of the vertices (n_nodes, 3).
#     panels : list
#         List with the vertices number for each panel.
#     plotter : pv.Plotter
#         The PyVista plotter to add the panels to.
#     color : str
#         Face color of the panels.
#     opacity : float
#         Opacity of the panels.
#     edge_color : str
#         Edge color of the panels.
#     line_width : float
#         Line width of the panel edges.
#     '''
#     for panel in panels:
#         # Create mesh from panel vertices
#         panel_points = nodes[panel]

#         # Define the faces (the first value is the number of points in the face)
#         face = [len(panel)] + list(range(len(panel)))
        
#         # Create a PolyData object for the panel
#         panel_mesh = pv.PolyData(panel_points)
#         panel_mesh.faces = np.hstack(face).astype(int)
        
#         # Add the panel to the plotter
#         plotter.add_mesh(panel_mesh, color=color, opacity=opacity, edge_color=edge_color, line_width=line_width)

# if __name__ == "__main__":
#     # Example data
#     nodes = np.array([
#         [0.000, 0, 0],
#         [0.707, 0, 0.707],
#         [1.414, 0, 0],
#         [0.000, 1, 0],
#         [0.707, 1, 0.707],
#         [1.414, 1, 0.707]
#     ])
    
#     panels = [
#         [0, 1, 4, 3],
#         [1, 2, 4],
#         [2, 5, 4]
#     ]

#     truss = {'Node': nodes, 'Trigl': None}
#     angles = {'Panel': panels}
#     U_his = np.cumsum(np.random.randn(nodes.shape[0] * 3, 20).reshape(nodes.shape[0] * 3, 20), axis=1) * 0.01

#     VisualFold(U_his, truss, angles)



import numpy as np
import pyvista as pv
from plot_ori_panels import save_gif_PIL
import os

def VisualFold(U_his, truss, angles, LF_his=None):
    '''The function `VisualFold` generates a 3D animation of a truss structure undergoing deformation and
    saves it as a GIF file.
    
    Parameters
    ----------
    U_his
        U_his is a numpy array containing the history of nodal displacements over time in the simulation.
    Each column represents the nodal displacements at a specific time step.
    truss
        The `truss` parameter in the `VisualFold` function represents the truss structure in the simulation.
    angles
        The `angles` parameter represents the angles of the panels in the truss structure.
    LF_his
        LF_his is a NumPy array that represents the load history of the truss structure.
    '''
    
    Node = truss['Node']
    Panel = angles['Panel']

    # if LF_his is not None and len(LF_his.shape) > 1:
    #     LF_his = np.sum(LF_his, axis=1)

    files = []
    Node_history = []

    # Set up PyVista plotter with off_screen rendering (more reliable on non-GUI/limited OpenGL systems)
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter.add_title("3D Truss Deformation")
    plotter.show_axes()

    # Set a stable camera orientation to avoid VTK view-up parallel warnings
    plotter.set_viewup([0, 0, 1])
    plotter.camera_position = 'xy'

    # Open a gif (this creates the render window/context)
    plotter.open_gif("ori.gif",
        # loop=0,
        # fps=10,
        palettesize=256,
        subrectangles=False        )

    # Ensure a valid render context exists before writing frames
    plotter.render()


    mins = Node.min(axis=0)
    maxs = Node.max(axis=0)

    # Adjust limits to avoid resetting each loop
    plotter.set_scale(xscale=(maxs[0] - mins[0]), yscale=(maxs[1] - mins[1]), zscale=(maxs[2] - mins[2]))

    for i in range(U_his.shape[1]):
        U = U_his[:, i]
        Nodew = Node.copy()
        Nodew[:, 0] += U[::3]
        Nodew[:, 1] += U[1::3]
        Nodew[:, 2] += U[2::3]

        # Clear existing actors without closing the window
        plotter.clear_actors()

        # Plot the original structure
        plot_panels(Node, Panel, plotter, color="#d86a96", opacity=0.3, edge_color="#858585", line_width=0.5)
        
        # Plot the deformed structure
        plot_panels(Nodew, Panel, plotter, color="#d86a96", edge_color="#3c3c3c", line_width=1)

        # Write a frame. This triggers a render.
        plotter.write_frame()

        # Render the current frame
        plotter.render()
        
        # Take a screenshot and save it
        file = f"ori_{str(i).zfill(2)}.png"
        plotter.screenshot(file)
        files.append(file)
        Node_history.append(Nodew)

    # In off-screen mode we do not show an interactive window.
    # plotter.show(auto_close=False)


    # Show the plot once all frames are created
    plotter.close()

    # Save GIF
    save_gif_PIL("ori_anim.gif", files, fps=5, loop=0)

    # Clean up the files
    # for file in files:
    #     os.remove(file)

def plot_panels(nodes, panels, plotter, color="#d86a96", opacity=1.0, edge_color="#3c3c3c", line_width=1):
    '''Plot origami panels in 3D as a collection of polygons using PyVista.
    
    Parameters
    ----------
    nodes : ndarray, float
        Coordinates of the vertices (n_nodes, 3).
    panels : list
        List with the vertices number for each panel.
    plotter : pv.Plotter
        The PyVista plotter to add the panels to.
    color : str
        Face color of the panels.
    opacity : float
        Opacity of the panels.
    edge_color : str
        Edge color of the panels.
    line_width : float
        Line width of the panel edges.
    '''
    for panel in panels:
        # Create mesh from panel vertices
        panel_points = nodes[panel]

        # Define the faces (the first value is the number of points in the face)
        face = [len(panel)] + list(range(len(panel)))
        
        # Create a PolyData object for the panel
        panel_mesh = pv.PolyData(panel_points)
        panel_mesh.faces = np.hstack(face).astype(int)
        
        # Add the panel to the plotter
        plotter.add_mesh(panel_mesh, color=color, opacity=opacity, edge_color=edge_color, line_width=line_width)

if __name__ == "__main__":
    # Example data
    nodes = np.array([
        [0.000, 0, 0],
        [0.707, 0, 0.707],
        [1.414, 0, 0],
        [0.000, 1, 0],
        [0.707, 1, 0.707],
        [1.414, 1, 0.707]
    ])
    
    panels = [
        [0, 1, 4, 3],
        [1, 2, 4],
        [2, 5, 4]
    ]

    truss = {'Node': nodes, 'Trigl': None}
    angles = {'Panel': panels}
    U_his = np.cumsum(np.random.randn(nodes.shape[0] * 3, 20).reshape(nodes.shape[0] * 3, 20), axis=1) * 0.01

    VisualFold(U_his, truss, angles)
