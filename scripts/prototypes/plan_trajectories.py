import nibabel as nib
import numpy as np
import os
from skimage import morphology, segmentation, feature, measure
from scipy import ndimage
from scipy.spatial import distance
import pyvista as pv

# --- Parameters ---
DATA_PATH = "../../data/CTpreop.nii"
ENTRY_POINT = np.array([139, 45, 135])

THRESHOLD = 3820
MIN_DISTANCE = 8
OPENING_RADIUS = 2
TARGET_MARKERS = 5

def plan_trajectories():
    # 1. Load data and run the pipeline to get the 5 markers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_PATH)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    img = nib.load(data_path)
    data = img.get_fdata()
    print("Processing CT data to find targets...")

    # Segmentation
    binary = data > THRESHOLD
    opened = morphology.opening(binary, morphology.ball(OPENING_RADIUS))
    
    dist_map = ndimage.distance_transform_edt(opened)
    coords = feature.peak_local_max(dist_map, min_distance=MIN_DISTANCE, labels=opened)
    
    mask_peaks = np.zeros(dist_map.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers_img, _ = ndimage.label(mask_peaks)
    labels = segmentation.watershed(-dist_map, markers_img, mask=opened)

    # Centroids
    props = measure.regionprops(labels)
    current_points = [np.array(prop.centroid) for prop in props]

    # Merging to 5 targets by iteratively merging the closest points until we have only 5 left
    while len(current_points) > TARGET_MARKERS:
        dist_matrix = distance.cdist(current_points, current_points)
        np.fill_diagonal(dist_matrix, np.inf)
        idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        new_point = (current_points[idx1] + current_points[idx2]) / 2.0
        first = max(idx1, idx2)
        second = min(idx1, idx2)
        current_points.pop(first)
        current_points.pop(second)
        current_points.append(new_point)

    targets = np.array(current_points)

    # 2. 3D Visualization using PyVista
    plotter = pv.Plotter(title="Surgical Path Planning")
    
    # Add Box Context (Optional but helpful for orientation)
    # We use a threshold to show the phantom structure vaguely
    box_binary = (data > 665) & (data < 3850)
    grid = pv.ImageData(dimensions=box_binary.shape)
    grid.point_data["values"] = box_binary.flatten(order="F")
    box_mesh = grid.contour([0.5])
    plotter.add_mesh(box_mesh, color="cyan", opacity=0.15, label="Phantom/Box")

    # Add Target Markers (Red Spheres)
    target_cloud = pv.PolyData(targets)
    plotter.add_mesh(target_cloud.glyph(geom=pv.Sphere(), factor=12, scale=False), 
                     color="red", label="Target Beads")

    # Add Entry Point (Blue Sphere)
    entry_cloud = pv.PolyData(ENTRY_POINT.reshape(1, 3))
    plotter.add_mesh(entry_cloud.glyph(geom=pv.Sphere(), factor=15, scale=False), 
                     color="blue", label="Entry Point")

    # Add Trajectories (Lines)
    for target in targets:
        # Create a line from entry to target
        line = pv.Line(ENTRY_POINT, target)
        plotter.add_mesh(line, color="yellow", line_width=3)

    # Plot Settings
    plotter.add_axes()
    plotter.add_legend()
    print("Opening 3D visualization. Targets, Entry Point, and Paths are visible.")
    plotter.show()

if __name__ == "__main__":
    plan_trajectories()
