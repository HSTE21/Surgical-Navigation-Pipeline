import nibabel as nib
import numpy as np
import os
from skimage import morphology, segmentation, feature
from scipy import ndimage

# Parameters
DATA_PATH = "../../data/CTpreop.nii"
OUTPUT_DIR = "../../output/"
OUTPUT_FILE = "beads_mask.nii.gz"
THRESHOLD = 3820
MIN_DISTANCE = 8
OPENING_RADIUS = 2

def segment_beads():
    # 1. Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    img = nib.load(DATA_PATH)
    data = img.get_fdata()
    affine = img.affine
    print(f"Data loaded. Shape: {data.shape}")

    # 2. Thresholding
    # Beads have significantly higher intensity values than the surrounding tissue.
    # Applying a high threshold isolates these bright structures.
    binary = data > THRESHOLD
    print(f"Thresholding applied (>{THRESHOLD}).")

    # 3. Morphological Opening
    # Morphological opening (erosion followed by dilation) is used to remove small 
    # bright noise particles while preserving the larger structures (the beads).
    selem = morphology.ball(OPENING_RADIUS)
    opened = morphology.opening(binary, selem)
    print(f"Morphological opening applied (radius={OPENING_RADIUS}).")

    # 4. Watershed Segmentation
    # Beads are often located close together and might appear as a single connected component.
    # We use a distance transform to find the center of each bead and then apply
    # the watershed algorithm to separate these touching objects into individual labels.
    distance = ndimage.distance_transform_edt(opened)
    coords = feature.peak_local_max(distance, min_distance=MIN_DISTANCE, labels=opened)
    
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask_peaks)
    
    labels = segmentation.watershed(-distance, markers, mask=opened)
    print(f"Watershed complete. Found {labels.max()} objects.")

    # 5. Save the result
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    nib.save(nib.Nifti1Image(labels.astype(np.uint16), affine), out_path)
    print(f"Saved result to {out_path}")

if __name__ == "__main__":
    segment_beads()
