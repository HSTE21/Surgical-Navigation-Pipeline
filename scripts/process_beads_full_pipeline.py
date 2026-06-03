import nibabel as nib
import numpy as np
import os
from skimage import morphology, segmentation, feature, measure
from scipy import ndimage
from scipy.spatial import distance

# --- Parameters ---
DATA_PATH = "../data/CTpreop.nii"
OUTPUT_DIR = "../output/"
OUTPUT_MASK = "beads_mask.nii.gz"

THRESHOLD = 3820
MIN_DISTANCE = 8
OPENING_RADIUS = 2
TARGET_MARKERS = 5

def process_markers():
    # 1. Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    img = nib.load(data_path)
    data = img.get_fdata()
    affine = img.affine
    print(f"Data loaded. Shape: {data.shape}")

    # 2. Segmentation Pipeline
    # Rationale: Thresholding isolates bright beads, Opening removes noise.
    binary = data > THRESHOLD
    selem = morphology.ball(OPENING_RADIUS)
    opened = morphology.opening(binary, selem)
    
    # Rationale: Watershed separates touching beads using distance transform.
    dist_map = ndimage.distance_transform_edt(opened)
    coords = feature.peak_local_max(dist_map, min_distance=MIN_DISTANCE, labels=opened)
    
    mask_peaks = np.zeros(dist_map.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers_img, _ = ndimage.label(mask_peaks)
    
    labels = segmentation.watershed(-dist_map, markers_img, mask=opened)
    print(f"Segmentation complete. Found {labels.max()} initial beads.")

    # 3. Calculate initial centroids (averaging voxel coordinates)
    # Rationale: Keep only the 10 largest components (5 markers * 2 beads) to remove noise.
    props = measure.regionprops(labels)
    # Sort by volume (area) and take the top 10
    props = sorted(props, key=lambda r: r.area, reverse=True)[:10]
    current_points = [np.array(prop.centroid) for prop in props]

    # 4. Iterative Merging Logic

    # Rationale: Merge closest pairs until we reach the 5 physical markers.
    while len(current_points) > TARGET_MARKERS:
        dist_matrix = distance.cdist(current_points, current_points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find closest pair
        idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        
        # Merge by averaging
        new_point = (current_points[idx1] + current_points[idx2]) / 2.0
        
        # Remove old, add new
        first = max(idx1, idx2)
        second = min(idx1, idx2)
        current_points.pop(first)
        current_points.pop(second)
        current_points.append(new_point)

    # 5. Output Results
    print(f"\nFinal {len(current_points)} markers reached (Pixel Coordinates):")
    print("-" * 45)
    print(f"{'Marker':<8} | {'X':>8} | {'Y':>8} | {'Z':>8}")
    print("-" * 45)
    for i, p in enumerate(current_points):
        print(f"{i+1:<8} | {p[0]:8.2f} | {p[1]:8.2f} | {p[2]:8.2f}")

    # Save mask for verification
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_path = os.path.join(output_dir, OUTPUT_MASK)
    nib.save(nib.Nifti1Image(labels.astype(np.uint16), affine), out_path)
    print(f"\nMask saved to: {out_path}")

    return current_points

if __name__ == "__main__":
    process_markers()
