import sys
import os
import json
import time
import numpy as np
import pyigtl
import nibabel as nib
import warnings
from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor
import pyvista as pv
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=UserWarning)

PROBE_OFFSETS = {
    "Probe_1": np.array([-128.08206577, -11.60711977, 5.77805026]),
    "Probe_2": np.array([-109.69918709, -4.54499096, 4.47161016])
}

class NDIValidationGUI(QtWidgets.QMainWindow):
    def __init__(self, host_ip="127.0.0.1", port=18944, probe_type="Probe_1"):
        super().__init__()
        self.setWindowTitle("NDI Intra-operative Validation")
        self.resize(1200, 800)

        self.host_ip = host_ip
        self.port = port
        self.probe_type = probe_type
        self.probe_offset = PROBE_OFFSETS.get(probe_type, np.zeros(3))
        
        try:
            self.client = pyigtl.OpenIGTLinkClient(host_ip, port)
        except Exception as e:
            print(f"Connection error: {e}")

        self.ndi_points = []
        self.target_points = []
        self.registration_matrix = None
        
        self.current_step = -1
        self.corners_voxels = []
        
        self.pointer_actor = None
        self.highlight_actor = None
        self.trajectory_actors = []
        
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QHBoxLayout(self.central_widget)

        self.sidebar = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.sidebar, 1)

        self.status_label = QtWidgets.QLabel("Status: Initializing...")
        self.sidebar.addWidget(self.status_label)
        self.instruction_label = QtWidgets.QLabel("Loading data...")
        self.sidebar.addWidget(self.instruction_label)

        self.next_btn = QtWidgets.QPushButton("CAPTURE POINT (Space)")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.on_next_clicked)
        self.sidebar.addWidget(self.next_btn)
        
        self.shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        self.shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut.activated.connect(self.on_next_clicked)

        self.sidebar.addStretch()

        self.plotter = QtInteractor(self.central_widget)
        self.layout.addWidget(self.plotter.interactor, 4)
        
        self.load_pipeline_data()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ndi)
        self.timer.start(16) # ~60Hz 

    def load_pipeline_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
        json_path = os.path.join(repo_root, "output", "transformed_coords.json")
        img_path = os.path.join(repo_root, "output", "CTpreop_registered.nii.gz")

        if not os.path.exists(json_path):
            self.status_label.setText("Error: transformed_coords.json not found!")
            return

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.corners_voxels = np.array(data["corners"])
            self.entry_voxel = np.array(data["entry"])
            self.beads_voxels = np.array(data["beads"])

            self.plotter.set_background("black")
            
            if os.path.exists(img_path):
                img = nib.load(img_path)
                data_img = img.get_fdata()
                mask = (data_img > 665) & (data_img < 3850)
                grid = pv.ImageData(dimensions=mask.shape)
                grid.point_data["values"] = mask.flatten(order="F")
                mesh = grid.contour([0.5])
                self.plotter.add_mesh(mesh, color="cyan", opacity=0.1, pickable=False)

            if len(self.beads_voxels) > 0:
                bead_cloud = pv.PolyData(self.beads_voxels)
                self.plotter.add_mesh(bead_cloud.glyph(geom=pv.Sphere(), factor=8, orient=False), color="red")
            
            for pt in self.corners_voxels:
                self.plotter.add_mesh(pv.Sphere(center=pt, radius=3), color="white", opacity=0.3)

            self.trajectory_actors = []
            for target in self.beads_voxels:
                line = pv.Line(self.entry_voxel, target)
                actor = self.plotter.add_mesh(line, color="yellow", line_width=3)
                self.trajectory_actors.append((line, actor, target))

            self.current_step = 0
            self.update_ui_state()
            self.plotter.reset_camera()
        except Exception as e:
            print(f"Error loading: {e}")

    def update_ui_state(self):
        if self.current_step < len(self.corners_voxels):
            self.instruction_label.setText(f"TOUCH CORNER {self.current_step + 1}")
            self.next_btn.setEnabled(True)
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor)
            pt = self.corners_voxels[self.current_step]
            self.highlight_actor = self.plotter.add_mesh(pv.Sphere(center=pt, radius=7), color="lime", opacity=0.6)
        else:
            self.instruction_label.setText("LIVE NAVIGATION")
            self.next_btn.hide()
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor)
                self.highlight_actor = None

    def on_next_clicked(self):
        ptr_data = self.get_current_ndi_data()
        if ptr_data is not None:
            tip_pos, _ = ptr_data
            self.ndi_points.append(tip_pos)
            self.target_points.append(self.corners_voxels[self.current_step])
            self.current_step += 1
            if self.current_step == len(self.corners_voxels):
                self.compute_registration()
            self.update_ui_state()

    def get_current_ndi_data(self):
        messages = self.client.get_latest_messages()
        if not messages:
            return None
        latest_msg = messages[-1]
            
        if isinstance(latest_msg, pyigtl.TransformMessage):
            mat = latest_msg.matrix
            pos = mat[:3, 3]
            rot_mat = mat[:3, :3]
            tip = rot_mat @ self.probe_offset + pos
            return tip, rot_mat
        elif isinstance(latest_msg, pyigtl.PositionMessage):
            pos = latest_msg.position
            quat = latest_msg.quaternion
            rot_mat = R.from_quat(quat).as_matrix()
            tip = rot_mat @ self.probe_offset + pos
            return tip, rot_mat
        return None

    def compute_registration(self):
        P, Q = np.array(self.ndi_points), np.array(self.target_points)
        cP, cQ = np.mean(P, axis=0), np.mean(Q, axis=0)
        H = (P - cP).T @ (Q - cQ)
        U, S, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T
        if np.linalg.det(R_mat) < 0:
            Vt[2,:] *= -1
            R_mat = Vt.T @ U.T
        self.registration_matrix = (R_mat, cQ - R_mat @ cP)

    def update_ndi(self):
        ptr_data = self.get_current_ndi_data()
        if ptr_data is not None:
            tip, rot_mat = ptr_data
            if self.registration_matrix:
                R_reg, t_reg = self.registration_matrix
                display_tip = R_reg @ tip + t_reg
                display_rot_mat = R_reg @ rot_mat
            else:
                display_tip, display_rot_mat = tip, rot_mat

            trans_mat = np.eye(4)
            trans_mat[:3, :3] = display_rot_mat
            trans_mat[:3, 3] = display_tip

            if self.pointer_actor is None:
                shaft_len = 160.0
                shaft = pv.Cylinder(center=(0, 0, -shaft_len/2), direction=(0, 0, 1), radius=2.5, height=shaft_len)
                self.pointer_actor = self.plotter.add_mesh(shaft, color="orange", opacity=1.0)
            
            # Use user_matrix for high-performance updates
            self.pointer_actor.user_matrix = trans_mat
            
            if self.registration_matrix:
                min_dist = 999
                for _, actor, target in self.trajectory_actors:
                    v, w = target - self.entry_voxel, display_tip - self.entry_voxel
                    c1, c2 = np.dot(w, v), np.dot(v, v)
                    if c1 <= 0: d = np.linalg.norm(display_tip - self.entry_voxel)
                    elif c2 <= c1: d = np.linalg.norm(display_tip - target)
                    else: d = np.linalg.norm(display_tip - (self.entry_voxel + (c1 / c2) * v))
                    min_dist = min(min_dist, d)
                    actor.prop.color = "green" if d < 2.0 else "yellow"
                    actor.prop.line_width = 8 if d < 2.0 else 3
                self.status_label.setText(f"Dist to Plan: {min_dist:.1f} mm")
            else:
                self.status_label.setText(f"Raw NDI Tip: {display_tip[0]:.0f}, {display_tip[1]:.0f}, {display_tip[2]:.0f}")
        else:
            if not self.registration_matrix:
                self.status_label.setText("Status: No NDI signal")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    window = NDIValidationGUI(host_ip=ip)
    window.show()
    sys.exit(app.exec_())
