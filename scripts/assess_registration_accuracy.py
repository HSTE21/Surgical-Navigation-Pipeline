import numpy as np
import nibabel as nib
import os
import json
import matplotlib.pyplot as plt
from skimage import morphology, segmentation, feature, measure
from scipy import ndimage
from scipy.spatial import distance, KDTree

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTRAOP_PATH = os.path.join(SCRIPT_DIR, "../data/CTintraop.nii")
TRANSFORMED_JSON_PATH = os.path.join(SCRIPT_DIR, "../output/transformed_coords.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output/")

# --- Constants ---
THRESHOLD_BEADS = 3820
MIN_DISTANCE_BEADS = 8
OPENING_RADIUS_BEADS = 2

THRESHOLD_BOX_MIN = 665
THRESHOLD_BOX_MAX = 3850

def detect_beads_intra(path, boundaries):
    """Detect beads within box boundaries."""
    xmin, xmax, ymin, ymax, zmin, zmax = boundaries
    margin = 5
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    binary = data > THRESHOLD_BEADS
    
    spatial_mask = np.zeros_like(binary, dtype=bool)
    spatial_mask[max(0, xmin-margin):min(data.shape[0], xmax+margin),
                 max(0, ymin-margin):min(data.shape[1], ymax+margin),
                 max(0, zmin-margin):min(data.shape[2], zmax+margin)] = True
    binary = binary & spatial_mask
    
    opened = morphology.opening(binary, morphology.ball(OPENING_RADIUS_BEADS))
    dist_map = ndimage.distance_transform_edt(opened)
    coords = feature.peak_local_max(dist_map, min_distance=MIN_DISTANCE_BEADS, labels=opened)
    mask_peaks = np.zeros(dist_map.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask_peaks)
    labels = segmentation.watershed(-dist_map, markers, mask=opened)
    
    # Save mask
    out_mask_path = os.path.join(OUTPUT_DIR, "beads_mask_intraop.nii.gz")
    nib.save(nib.Nifti1Image(labels.astype(np.uint16), affine), out_mask_path)
    
    props = measure.regionprops(labels)
    # Rationale: Keep only the 10 largest components (5 markers * 2 beads) to remove noise.
    props = sorted(props, key=lambda r: r.area, reverse=True)[:10]
    points = [np.array(p.centroid) for p in props]
    
    while len(points) > 5:
        dist_mat = distance.cdist(points, points)
        np.fill_diagonal(dist_mat, np.inf)
        idx1, idx2 = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        new_p = (points[idx1] + points[idx2]) / 2.0
        points.pop(max(idx1, idx2)); points.pop(min(idx1, idx2))
        points.append(new_p)
    return points

def detect_corners_intra(path):
    img = nib.load(path)
    data = img.get_fdata()
    binary = (data >= THRESHOLD_BOX_MIN) & (data <= THRESHOLD_BOX_MAX)
    hist_x = np.sum(binary, axis=(1, 2))
    hist_y = np.sum(binary, axis=(0, 2))
    hist_z = np.sum(binary, axis=(0, 1))
    
    def find_bounds(hist):
        thresh = np.max(hist) * 0.05
        indices = np.where(hist > thresh)[0]
        return indices[0], indices[-1]
    
    bounds = (find_bounds(hist_x), find_bounds(hist_y), find_bounds(hist_z))
    xmin, xmax = bounds[0]; ymin, ymax = bounds[1]; zmin, zmax = bounds[2]
    
    corners_calc = []
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            for z in [zmin, zmax]:
                corners_calc.append(np.array([x, y, z]))
    
    box_voxels = np.argwhere(binary)
    tree = KDTree(box_voxels)
    _, nearest_indices = tree.query(corners_calc)
    return box_voxels[nearest_indices], (xmin, xmax, ymin, ymax, zmin, zmax), binary

def assess_accuracy():
    # 1. Detect Ground Truth
    actual_corners, boundaries, box_binary = detect_corners_intra(INTRAOP_PATH)
    actual_beads = detect_beads_intra(INTRAOP_PATH, boundaries)
    
    # 2. Load Prediction
    with open(TRANSFORMED_JSON_PATH, 'r') as f:
        predicted_data = json.load(f)
    pred_beads = np.array(predicted_data["beads"])
    pred_corners = np.array(predicted_data["corners"])
    
    # 3. Matching
    tree_beads = KDTree(actual_beads)
    _, idx_beads = tree_beads.query(pred_beads)
    matched_beads_actual = np.array(actual_beads)[idx_beads]
    
    tree_corners = KDTree(actual_corners)
    _, idx_corners = tree_corners.query(pred_corners)
    matched_corners_actual = np.array(actual_corners)[idx_corners]

    # 4. 3D Visualization for Diagnosis
    try:
        import pyvista as pv
        print("\nPreparing 3D diagnosis plot...")
        plotter = pv.Plotter(title="Registration Accuracy: Prediction vs Reality")
        
        # Show Box Mesh for context
        grid = pv.ImageData(dimensions=box_binary.shape)
        grid.point_data["values"] = box_binary.flatten(order="F")
        mesh = grid.contour([0.5])
        plotter.add_mesh(mesh, color="cyan", opacity=0.1, label="Intra-op Box")
        
        # Predicted Beads (Red) - Where we think they are
        plotter.add_mesh(pv.PolyData(pred_beads).glyph(geom=pv.Sphere(), factor=10, scale=False), 
                         color="red", label="Predicted (Plan)")
        
        # Actual Detected Beads (Green) - Where they really are
        plotter.add_mesh(pv.PolyData(actual_beads).glyph(geom=pv.Sphere(), factor=10, scale=False), 
                         color="green", label="Actual (Ground Truth)")

        # Draw lines between matched points to visualize the error vectors
        for i in range(len(pred_beads)):
            line = pv.Line(pred_beads[i], matched_beads_actual[i])
            plotter.add_mesh(line, color="yellow", line_width=3)

        plotter.add_axes()
        plotter.add_legend()
        print("Opening 3D Plot. RED = Plan, GREEN = Reality. Yellow lines = Error magnitude.")
        plotter.show()
    except ImportError:
        print("\nPyVista not found. Skipping 3D visualization.")

    # 5. Error Table (Voxel Space)
    print("\n" + "="*75)
    print("REGISTRATION ERROR ASSESSMENT (Voxel Space)")
    print("="*75)
    print(f"{'Point Name':<20} | {'dX':>10} | {'dY':>10} | {'dZ':>10} | {'Total':>12}")
    print("-" * 75)
    
    def print_err(name, p_act, p_pred):
        d = np.abs(p_act - p_pred)
        total = np.linalg.norm(p_act - p_pred)
        print(f"{name:<20} | {d[0]:10.2f} | {d[1]:10.2f} | {d[2]:10.2f} | {total:12.2f}")

    # Entry Point Error Calculation
    # Ground Truth provided by user: [144, 51, 140]
    actual_entry = np.array([144, 51, 140])
    pred_entry = np.array(predicted_data["entry"])
    print_err("Entry Point", actual_entry, pred_entry)
    print("-" * 75)

    for i in range(len(pred_beads)):
        p_act = matched_beads_actual[i]
        p_pred = pred_beads[i]
        print_err(f"Bead {i+1}", p_act, p_pred)

    for i in range(len(pred_corners)):
        p_act = matched_corners_actual[i]
        p_pred = pred_corners[i]
        print_err(f"Corner {i+1}", p_act, p_pred)
    
    print("="*75)

if __name__ == "__main__":
    assess_accuracy()
