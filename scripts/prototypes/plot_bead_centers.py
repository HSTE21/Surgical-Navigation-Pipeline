import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from pathlib import Path

try:
    import pyvista as pv
except ImportError:
    pv = None

# Parameters
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MASK_PATH = PROJECT_ROOT / "output" / "beads_mask.nii.gz"

def plot_bead_centers():
    # 1. Load the segmentation mask
    if not os.path.exists(MASK_PATH):
        print(f"Error: {MASK_PATH} not found.")
        print("Please ensure you have run 'segment_beads.py' and the output file exists.")
        return

    img = nib.load(MASK_PATH)
    mask_data = img.get_fdata().astype(np.uint16)
    print(f"Mask loaded. Shape: {mask_data.shape} (X, Y, Z)")

    # 2. Calculate centroids (averaging X, Y, Z pixel coordinates for each label)
    # regionprops calculates the centroid as the arithmetic mean of all pixel indices in the label.
    props = measure.regionprops(mask_data)

    centroids = []
    print("\nCalculated Bead Centers (Pixel/Voxel Coordinates):")
    print("-" * 45)
    print(f"{'Label ID':<10} | {'X':>8} | {'Y':>8} | {'Z':>8}")
    print("-" * 45)

    for prop in props:
        # Centroid is returned as (X, Y, Z) based on the input array dimensions
        c = prop.centroid
        centroids.append((prop.label, c))
        print(f"{prop.label:<10} | {c[0]:8.2f} | {c[1]:8.2f} | {c[2]:8.2f}")

    def plot_centroid_points_pyvista(centroids):
        if pv is None:
            print("PyVista not available. Install it with `pip install pyvista` to use the point visualization.")
            return

        point_coords = np.array([c for _, c in centroids])
        label_ids = np.array([label_id for label_id, _ in centroids], dtype=int)
        point_cloud = pv.PolyData(point_coords)
        point_cloud["Label"] = label_ids

        p = pv.Plotter()
        p.add_points(point_cloud,
                     render_points_as_spheres=True,
                     point_size=15,
                     scalars="Label",
                     cmap="nipy_spectral",
                     opacity=1.0)
        p.add_axes()
        p.show_grid()
        p.show(title=f"Bead Centroids in PyVista (N={len(centroids)})")

    plot_centroid_points_pyvista(centroids)

    # 3. 3D Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    n_labels = len(props)
    cmap = plt.get_cmap('nipy_spectral', n_labels + 1)

    for i, (label_id, centroid) in enumerate(centroids):
        # Create surface for the bead
        bead_mask = (mask_data == label_id)
        try:
            verts, faces, normals, values = measure.marching_cubes(bead_mask, 0.5)
            
            # Color based on label
            color = cmap(i + 1)
            
            # Plot the bead surface
            mesh = Poly3DCollection(verts[faces], facecolor=color, edgecolor='k', linewidths=0.1, alpha=0.5)
            ax.add_collection3d(mesh)
            
            # Plot the centroid as a black dot
            ax.scatter(centroid[0], centroid[1], centroid[2], color='black', s=50, label=f'Center {label_id}' if i==0 else "")
            
        except Exception as e:
            print(f"Label {label_id}: Error rendering surface: {e}")

    # Plot settings
    ax.set_xlim(0, mask_data.shape[0])
    ax.set_ylim(0, mask_data.shape[1])
    ax.set_zlim(0, mask_data.shape[2])
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (voxels)')
    ax.set_title(f'Bead Segmentation with Calculated Centroids (N={n_labels})')
    
    print("\nShowing plot... (Close the plot window to finish)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_bead_centers()
