import itk
import numpy as np
import nibabel as nib
import os
import sys
import json

# Import functions from existing scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from process_beads_full_pipeline import process_markers
from get_box_corners_v2 import get_box_corners_histogram

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_IMAGE_PATH = os.path.join(SCRIPT_DIR, "../data/CTintraop.nii")
MOVING_IMAGE_PATH = os.path.join(SCRIPT_DIR, "../data/CTpreop.nii")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../output/")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Configuration ---
ENTRY_POINT_VOXEL = np.array([139, 45, 135])

def voxel_to_physical(points_voxel, itk_image):
    """Use ITK's native metadata-aware conversion."""
    points_phys = []
    for p in points_voxel:
        idx = itk.Index[3]()
        idx[0], idx[1], idx[2] = int(round(float(p[0]))), int(round(float(p[1]))), int(round(float(p[2])))
        pt = itk_image.TransformIndexToPhysicalPoint(idx)
        points_phys.append(np.array([pt[0], pt[1], pt[2]]))
    return points_phys

def physical_to_voxel(points_phys, itk_image):
    """Use ITK's native metadata-aware conversion."""
    points_voxel = []
    for p in points_phys:
        pt = itk.Point[itk.D, 3](p)
        idx = itk_image.TransformPhysicalPointToContinuousIndex(pt)
        points_voxel.append(np.array([idx[0], idx[1], idx[2]]))
    return points_voxel

def parse_transformix_output(file_path):
    transformed_points = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        for line in f:
            if 'OutputPoint = [' in line:
                parts = line.split('OutputPoint = [')[1].split(']')[0].split()
                transformed_points.append(np.array([float(x) for x in parts]))
    return transformed_points

def run_registration_and_transform():
    # 1. Get Preop Points
    print("Extracting Preoperative points...")
    beads_preop = process_markers()
    # get_box_corners_histogram returns a tuple; we only need the first element (corners_voxel)
    corners_preop_result = get_box_corners_histogram()
    corners_preop = corners_preop_result[0]
    
    all_preop_voxels = [ENTRY_POINT_VOXEL] + beads_preop + list(corners_preop)

    print("Loading images...")
    img_preop_itk = itk.imread(MOVING_IMAGE_PATH, itk.F)
    img_intra_itk = itk.imread(FIXED_IMAGE_PATH, itk.F)

    # Common parameters for both runs
    def get_common_param_obj():
        obj = itk.ParameterObject.New()
        rigid_map = obj.GetDefaultParameterMap('rigid')
        rigid_map['NumberOfResolutions'] = ['5']
        rigid_map['MaximumNumberOfIterations'] = ['2000']
        rigid_map['NumberOfThreads'] = ['8']
        rigid_map['AutomaticTransformInitialization'] = ['true']
        rigid_map['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
        obj.AddParameterMap(rigid_map)

        bspline_map = obj.GetDefaultParameterMap('bspline')
        bspline_map['FinalGridSpacingInPhysicalUnits'] = ['10.0']
        bspline_map['NumberOfResolutions'] = ['4']
        bspline_map['MaximumNumberOfIterations'] = ['4000']
        bspline_map['NumberOfThreads'] = ['8']
        obj.AddParameterMap(bspline_map)
        return obj

    # --- STEP 2A: REGISTRATION FOR IMAGE (Pre-op -> Intra-op) ---
    # Goal: Warp the Pre-op plan onto the Intra-op scan for visualization.
    print("\n[1/2] Running registration for Image Warping (Fixed=Intra, Moving=Pre)...")
    param_obj_img = get_common_param_obj()
    result_image, _ = itk.elastix_registration_method(
        img_intra_itk, img_preop_itk, parameter_object=param_obj_img)

    # Save the warped plan image
    output_img_path = os.path.join(OUTPUT_DIR, "CTpreop_registered.nii.gz")
    itk.imwrite(result_image, output_img_path)
    print(f"Warped Pre-op plan saved to {output_img_path}")

    # --- STEP 2B: REGISTRATION FOR POINTS (Pre-op -> Intra-op) ---
    # Goal: Use Transformix's standard SetFixedPointSetFileName by making Pre-op the Fixed image.
    print("\n[2/2] Running registration for Point Transformation (Fixed=Pre, Moving=Intra)...")
    param_obj_pts = get_common_param_obj()
    _, result_transform_parameters = itk.elastix_registration_method(
        img_preop_itk, img_intra_itk, parameter_object=param_obj_pts)

    # 3. Transform Points
    print("Transforming points...")
    all_preop_phys = voxel_to_physical(all_preop_voxels, img_preop_itk)
    
    point_set_file = os.path.join(OUTPUT_DIR, "preop_points.txt")
    with open(point_set_file, "w") as f:
        f.write(f"point\n{len(all_preop_phys)}\n")
        for p in all_preop_phys:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    transformix_object = itk.TransformixFilter.New(img_intra_itk)
    transformix_object.SetTransformParameterObject(result_transform_parameters)
    # Using standard SetFixedPointSetFileName on the correct registration direction
    transformix_object.SetFixedPointSetFileName(point_set_file)
    transformix_object.SetOutputDirectory(OUTPUT_DIR)
    transformix_object.Update()

    # 4. Convert back to Voxel space of CTintraop (de moving_image in Run 2)
    transformed_phys = parse_transformix_output(os.path.join(OUTPUT_DIR, "outputpoints.txt"))
    intraop_voxels = physical_to_voxel(transformed_phys, img_intra_itk)

    # 5. Export Results (Dynamic Slicing)
    n_beads = len(beads_preop)
    n_corners = len(list(corners_preop))
    
    transformed_data = {
        "entry": intraop_voxels[0].tolist(),
        "beads": [p.tolist() for p in intraop_voxels[1 : 1 + n_beads]],
        "corners": [p.tolist() for p in intraop_voxels[1 + n_beads : 1 + n_beads + n_corners]]
    }
    
    json_path = os.path.join(OUTPUT_DIR, "transformed_coords.json")
    with open(json_path, 'w') as f:
        json.dump(transformed_data, f, indent=4)
    print(f"\nFinal Results saved to {json_path}")

if __name__ == "__main__":
    run_registration_and_transform()
