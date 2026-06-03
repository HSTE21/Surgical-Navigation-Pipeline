import numpy as np
import nibabel as nib
import os
import sys
import json

# Ensure we can import from the scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Import necessary functions to get original Pre-op points
from process_beads_full_pipeline import process_markers
from get_box_corners_v2 import get_box_corners_histogram

# --- Paths ---
JSON_COORDS_PATH = os.path.join(SCRIPT_DIR, "../../output/transformed_coords.json")

# --- Configuration ---
ENTRY_POINT_VOXEL_PREOP = np.array([139, 45, 135])

def calculate_displacement_errors():
    # 1. Load Transformed (Intra-op) Data from JSON
    if not os.path.exists(JSON_COORDS_PATH):
        print(f"Error: {JSON_COORDS_PATH} not found. Please run 'register_and_transform.py' first.")
        return

    with open(JSON_COORDS_PATH, 'r') as f:
        data_intra = json.load(f)
    
    # 2. Get Original (Pre-op) Data
    print("Extracting original Pre-operative points...")
    beads_pre = process_markers()
    corners_pre = get_box_corners_histogram()
    
    # Organize Pre-op points in the same order as JSON
    pre_points = {
        "Entry Point": ENTRY_POINT_VOXEL_PREOP,
        "Bead 1": beads_pre[0],
        "Bead 2": beads_pre[1],
        "Bead 3": beads_pre[2],
        "Bead 4": beads_pre[3],
        "Bead 5": beads_pre[4]
    }
    # Add box corners as "Pivots"
    for i, p in enumerate(corners_pre):
        pre_points[f"Pivot {i+1}"] = p

    # Organize Intra-op points from JSON
    intra_points = {
        "Entry Point": np.array(data_intra["entry"]),
        "Bead 1": np.array(data_intra["beads"][0]),
        "Bead 2": np.array(data_intra["beads"][1]),
        "Bead 3": np.array(data_intra[ "beads"][2]),
        "Bead 4": np.array(data_intra["beads"][3]),
        "Bead 5": np.array(data_intra["beads"][4])
    }
    for i, p in enumerate(data_intra["corners"]):
        intra_points[f"Pivot {i+1}"] = np.array(p)

    # 3. Calculate and Print Errors (Absolute Differences)
    print("\n" + "="*70)
    print(f"{'Point Name':<20} | {'dX':>10} | {'dY':>10} | {'dZ':>10} | {'Total':>10}")
    print("-" * 70)

    for name in pre_points.keys():
        p_pre = pre_points[name]
        p_intra = intra_points[name]
        
        # Calculate absolute difference per axis
        diff = np.abs(p_intra - p_pre)
        # Calculate Euclidean distance (magnitude of displacement)
        total_dist = np.linalg.norm(p_intra - p_pre)
        
        print(f"{name:<20} | {diff[0]:10.2f} | {diff[1]:10.2f} | {diff[2]:10.2f} | {total_dist:10.2f}")

    print("="*70)
    print("Note: All values are in Pixel/Voxel units.")
    print("dX, dY, dZ are absolute differences per axis.")
    print("Total is the 3D Euclidean displacement (magnitude of the shift).")

if __name__ == "__main__":
    calculate_displacement_errors()
