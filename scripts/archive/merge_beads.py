import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from scipy.spatial import distance

# Robust path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_PATH = os.path.join(SCRIPT_DIR, "..", "..", "output", "beads_mask.nii.gz")

def merge_beads_to_five():
    # 1. Load the segmentation mask
    if not os.path.exists(MASK_PATH):
        print(f"Error: {MASK_PATH} not found. Please run the segmentation script first.")
        return

    img = nib.load(MASK_PATH)
    mask_data = img.get_fdata().astype(np.uint16)
    print(f"Mask loaded. Shape: {mask_data.shape}")

    # 2. Calculate initial centroids for all detected beads (e.g., 10 beads)
    props = measure.regionprops(mask_data)
    # We store centroids as a list of numpy arrays for easy calculation
    current_points = [np.array(prop.centroid) for prop in props]
    print(f"Initially found {len(current_points)} beads.")

    # 3. Iteratively merge the two closest points until only 5 remain
    # This addresses the case where a single marker is detected as two separate beads.
    while len(current_points) > 5:
        # Calculate distance matrix between all current points
        dist_matrix = distance.cdist(current_points, current_points)
        
        # We want to find the minimum distance that is not zero (diagonal)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find indices of the two closest points
        idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        
        # Calculate the average (midpoint) of these two points
        p1 = current_points[idx1]
        p2 = current_points[idx2]
        new_point = (p1 + p2) / 2.0
        
        print(f"Merging points at distance {np.min(dist_matrix):.2f}")
        
        # Update the list: remove the two old points and add the new averaged point
        # (Remove higher index first to avoid index shifting issues)
        first = max(idx1, idx2)
        second = min(idx1, idx2)
        current_points.pop(first)
        current_points.pop(second)
        current_points.append(new_point)

    print(f"\nFinal {len(current_points)} markers reached.")
    print("-" * 45)
    print(f"{'Marker ID':<10} | {'X':>8} | {'Y':>8} | {'Z':>8}")
    print("-" * 45)
    for i, p in enumerate(current_points):
        print(f"{i+1:<10} | {p[0]:8.2f} | {p[1]:8.2f} | {p[2]:8.2f}")

    # 4. 3D Visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original beads (from mask) as transparent surfaces
    cmap = plt.get_cmap('nipy_spectral', len(props) + 1)
    for i, prop in enumerate(props):
        bead_mask = (mask_data == prop.label)
        try:
            verts, faces, _, _ = measure.marching_cubes(bead_mask, 0.5)
            mesh = Poly3DCollection(verts[faces], facecolor=cmap(i+1), edgecolor='k', linewidths=0.1, alpha=0.3)
            ax.add_collection3d(mesh)
        except:
            pass

    # Plot the 5 merged markers as large red dots
    final_points = np.array(current_points)
    ax.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], 
               color='red', s=100, label='Final 5 Markers', depthshade=False)

    ax.set_xlim(0, mask_data.shape[0])
    ax.set_ylim(0, mask_data.shape[1])
    ax.set_zlim(0, mask_data.shape[2])
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (pixels)')
    ax.set_title('Merged Markers (10 beads -> 5 points)')
    ax.legend()
    
    print("\nShowing plot...")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    merge_beads_to_five()
