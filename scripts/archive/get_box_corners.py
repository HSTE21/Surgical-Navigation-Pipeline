import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# --- Parameters ---
DATA_PATH = "../../data/CTpreop.nii"
THRESHOLD_MIN = 665
THRESHOLD_MAX = 3850

def get_box_corners():
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
    # Isolating the box based on the specified intensity range
    binary = (data >= THRESHOLD_MIN) & (data <= THRESHOLD_MAX)
    
    # --- ROBUST CLEANING ---
    # 1. Morphological Opening to remove small noise and break thin "bridges" 
    # to other structures (like the head or the CT table).
    print("Cleaning segmentation...")
    clean_binary = morphology.binary_opening(binary, morphology.ball(2))
    
    # 2. Keep only the largest connected component
    labels = measure.label(clean_binary)
    if labels.max() == 0:
        print("No structures found in this threshold range after cleaning.")
        return
        
    regions = measure.regionprops(labels)
    largest_region = max(regions, key=lambda r: r.area)
    box_mask = (labels == largest_region.label)
    print(f"Box isolated. Volume: {largest_region.area} voxels.")

    # 3. PCA to find Oriented Bounding Box (OBB)
    # Get all voxel coordinates of the box
    coords = np.argwhere(box_mask)
    
    # --- STATISTICAL OUTLIER REMOVAL ---
    # Remove points that are far from the main cluster to prevent "floating" corners.
    mean_init = np.mean(coords, axis=0)
    dist_to_mean = np.linalg.norm(coords - mean_init, axis=1)
    threshold_dist = np.percentile(dist_to_mean, 98) # Keep the 98% closest points
    coords = coords[dist_to_mean <= threshold_dist]
    
    # Calculate centroid and center the coordinates
    mean = np.mean(coords, axis=0)
    centered_coords = coords - mean
    
    # Covariance matrix and Eigen-decomposition
    cov = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvectors by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Project coordinates onto the principal axes
    projected = centered_coords @ eigenvectors
    
    # Robust Find: use tighter percentiles (1st and 99th)
    # This ignores the absolute extrema which are often noise/posts
    min_p = np.percentile(projected, 1, axis=0)
    max_p = np.percentile(projected, 99, axis=0)
    
    # Generate the 8 corners in the projected space
    corners_proj = np.array([
        [min_p[0], min_p[1], min_p[2]],
        [min_p[0], min_p[1], max_p[2]],
        [min_p[0], max_p[1], min_p[2]],
        [min_p[0], max_p[1], max_p[2]],
        [max_p[0], min_p[1], min_p[2]],
        [max_p[0], min_p[1], max_p[2]],
        [max_p[0], max_p[1], min_p[2]],
        [max_p[0], max_p[1], max_p[2]]
    ])
    
    # Transform corners back to the original voxel space
    corners_voxel = (corners_proj @ eigenvectors.T) + mean

    # 4. Results Output
    print("\nDetected Box Corners (Pixel Coordinates):")
    print("-" * 45)
    for i, corner in enumerate(corners_voxel):
        print(f"Corner {i+1}: X={corner[0]:.2f}, Y={corner[1]:.2f}, Z={corner[2]:.2f}")

    # 5. High-Performance 3D Visualization using PyVista (VTK-based)
    try:
        import pyvista as pv
        
        # Create a PyVista plotter
        plotter = pv.Plotter(title="Box Corners Verification")
        
        # Create the box mesh
        # We use the voxel coordinates to create a point cloud then a surface
        # Or more efficiently, use the mask directly with contouring
        grid = pv.ImageData(dimensions=box_mask.shape)
        grid.point_data["values"] = box_mask.flatten(order="F")
        mesh = grid.contour([0.5])
        
        # Add the mesh to the plotter
        plotter.add_mesh(mesh, color="cyan", opacity=0.3, label="Box Segmentation")
        
        # Add the corners as spheres
        corner_points = pv.PolyData(corners_voxel)
        plotter.add_mesh(corner_points.glyph(scale=5, geom=pv.Sphere()), color="red", label="Corners")
        
        # Add axes and labels
        plotter.add_axes()
        plotter.add_legend()
        
        print("\nOpening PyVista interactive window...")
        plotter.show()
        
    except ImportError:
        print("\nPyVista not found. For high-performance 3D plots, install it via:")
        print("pip install pyvista")
        print("\nFalling back to terminal output only.")

if __name__ == "__main__":
    get_box_corners()
