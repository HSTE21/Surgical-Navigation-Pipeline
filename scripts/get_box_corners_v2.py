import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Parameters ---
DATA_PATH = "../data/CTpreop.nii"
THRESHOLD_MIN = 665
THRESHOLD_MAX = 3850

def get_box_corners_histogram():
    # 1. Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    img = nib.load(data_path)
    data = img.get_fdata()
    print(f"Data loaded. Shape: {data.shape}")

    # 2. Segmentation
    binary = (data >= THRESHOLD_MIN) & (data <= THRESHOLD_MAX)
    
    # 3. Histogram Analysis (Voxel Density along axes)
    # We sum the binary mask along two axes to get the density along the third.
    hist_x = np.sum(binary, axis=(1, 2))
    hist_y = np.sum(binary, axis=(0, 2))
    hist_z = np.sum(binary, axis=(0, 1))

    # Helper to find boundaries based on a density threshold
    def find_boundaries(hist, label, threshold_ratio=0.01):
        # We look for where the density is at least 1% of the maximum
        # This ignores sparse noise and finds the "mass" of the walls.
        threshold = np.max(hist) * threshold_ratio
        indices = np.where(hist > threshold)[0]
        if len(indices) == 0:
            return 0, hist.shape[0]
        return indices[0], indices[-1]

    x_min, x_max = find_boundaries(hist_x, "X")
    y_min, y_max = find_boundaries(hist_y, "Y")
    z_min, z_max = find_boundaries(hist_z, "Z")

    corners_calculated = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ])

    # 4. Snapping: Project each corner to the nearest actual voxel in the box mask
    # This ensures corners sit exactly on the (ribbelige) surface of the box.
    print("Snapping corners to nearest voxels on the box surface...")
    box_voxels = np.argwhere(binary)
    
    from scipy.spatial import KDTree
    tree = KDTree(box_voxels)
    
    # Find the nearest voxel for each calculated corner
    _, nearest_indices = tree.query(corners_calculated)
    corners_voxel = box_voxels[nearest_indices]

    # 5. Results Output
    print("\nDetected Box Corners (Snapped to Nearest Voxels):")
    print("-" * 55)
    for i, corner in enumerate(corners_voxel):
        print(f"Corner {i+1}: X={corner[0]:.2f}, Y={corner[1]:.2f}, Z={corner[2]:.2f}")

    return corners_voxel, hist_x, hist_y, hist_z, x_min, x_max, y_min, y_max, z_min, z_max, binary

if __name__ == "__main__":
    result = get_box_corners_histogram()
    if result is None:
        raise SystemExit(1)

    corners_voxel, hist_x, hist_y, hist_z, x_min, x_max, y_min, y_max, z_min, z_max, binary = result
    
    # 5. Visualization: Histograms
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(hist_x, color='r')
    axs[0].axvline(x_min, color='k', linestyle='--')
    axs[0].axvline(x_max, color='k', linestyle='--')
    axs[0].set_title("Voxel Density along X-axis")
    
    axs[1].plot(hist_y, color='g')
    axs[1].axvline(y_min, color='k', linestyle='--')
    axs[1].axvline(y_max, color='k', linestyle='--')
    axs[1].set_title("Voxel Density along Y-axis")
    
    axs[2].plot(hist_z, color='b')
    axs[2].axvline(z_min, color='k', linestyle='--')
    axs[2].axvline(z_max, color='k', linestyle='--')
    axs[2].set_title("Voxel Density along Z-axis")
    
    plt.tight_layout()
    print("\nShowing Density Histograms...")
    plt.show()

    # 6. 3D Visualization using PyVista
    try:
        import pyvista as pv
        plotter = pv.Plotter(title="Histogram-Based Box Corners")
        
        # Create the mesh (downsample for speed if needed)
        grid = pv.ImageData(dimensions=binary.shape)
        grid.point_data["values"] = binary.flatten(order="F")
        mesh = grid.contour([0.5])
        
        plotter.add_mesh(mesh, color="cyan", opacity=0.3, label="Box Mask")
        
        # Add corners as spheres. Use 'factor' to control the size.
        corner_points = pv.PolyData(corners_voxel)
        plotter.add_mesh(corner_points.glyph(geom=pv.Sphere(), factor=10, scale=False), 
                         color="red", label="Corners")
        
        # Add a bounding box line for clarity
        outline = pv.Cube(center=((x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2),
                         x_length=(x_max-x_min), y_length=(y_max-y_min), z_length=(z_max-z_min))
        plotter.add_mesh(outline, color="yellow", style="wireframe", label="Calculated Box Bound")

        plotter.add_axes()
        plotter.add_legend()
        print("\nOpening PyVista 3D window...")
        plotter.show()
        
    except ImportError:
        print("\nPyVista not found. Install with 'pip install pyvista' for 3D verification.")
