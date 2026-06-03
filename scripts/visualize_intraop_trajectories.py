import numpy as np
import nibabel as nib
import os
import json
import pyvista as pv

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_IMAGE_PATH = os.path.join(SCRIPT_DIR, "../data/CTintraop.nii")
REGISTERED_IMAGE_PATH = os.path.join(SCRIPT_DIR, "../output/CTpreop_registered.nii.gz")
JSON_COORDS_PATH = os.path.join(SCRIPT_DIR, "../output/transformed_coords.json")

def visualize_intraop_trajectories():
    # 1. Verification of required files
    if not os.path.exists(REGISTERED_IMAGE_PATH) or not os.path.exists(JSON_COORDS_PATH):
        print("Error: Missing required files. Please run 'register_and_transform.py' first.")
        return

    # 2. Load Transformed Data from JSON (saved by registration script)
    print("Loading transformed coordinates and registered scan...")
    with open(JSON_COORDS_PATH, 'r') as f:
        data_coords = json.load(f)
    
    entry_intra = np.array(data_coords["entry"])
    beads_intra = np.array(data_coords["beads"])
    corners_intra = np.array(data_coords["corners"])

    # 3. Load Registered Pre-op Image (the warped image)
    # We use this to show the segmentation result in the new space
    img_reg = nib.load(REGISTERED_IMAGE_PATH)
    reg_data = img_reg.get_fdata()

    # 4. Visualization with PyVista
    print("Preparing 3D Visualization...")
    plotter = pv.Plotter(title="Final Intra-operative Validation")

    # A. Registered Pre-op Brain/Phantom (The actual warped plan)
    # Thresholding the registered image to show where the 'planned' anatomy lies now
    plan_mask = (reg_data > 665) & (reg_data < 3850)
    grid = pv.ImageData(dimensions=plan_mask.shape)
    grid.point_data["values"] = plan_mask.flatten(order="F")
    mesh = grid.contour([0.5])
    plotter.add_mesh(mesh, color="cyan", opacity=0.15, label="Warped Pre-op Plan")

    # B. Transformed Beads (Red Spheres)
    bead_cloud = pv.PolyData(beads_intra)
    plotter.add_mesh(bead_cloud.glyph(geom=pv.Sphere(), factor=10, scale=False), 
                     color="red", label="Transformed Beads")

    # C. Transformed Entry Point (Blue Sphere)
    entry_cloud = pv.PolyData(entry_intra.reshape(1, 3))
    plotter.add_mesh(entry_cloud.glyph(geom=pv.Sphere(), factor=12, scale=False), 
                     color="blue", label="Transformed Entry Point")

    # D. Transformed Box Corners (White Spheres)
    corner_cloud = pv.PolyData(corners_intra)
    plotter.add_mesh(corner_cloud.glyph(geom=pv.Sphere(), factor=8, scale=False), 
                     color="white", label="Transformed Box Corners")

    # E. Trajectories (Yellow Lines)
    for target in beads_intra:
        line = pv.Line(entry_intra, target)
        plotter.add_mesh(line, color="yellow", line_width=4)

    # Add axes and legend
    plotter.add_axes()
    plotter.add_legend()
    
    print("\nVisualizing the PROJECTED surgical plan.")
    print("The cyan structure is the REGISTERED pre-operative image.")
    print("The points and lines are the TRANSFORMED plan.")
    plotter.show()

if __name__ == "__main__":
    visualize_intraop_trajectories()
