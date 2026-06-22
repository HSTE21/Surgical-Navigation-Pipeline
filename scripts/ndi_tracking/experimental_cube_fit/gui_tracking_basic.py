# Cube implementation of the detected points for registration added.
# This script is a copy but with a registation method that snaps the detected points to a perfect cube. 
# It is put in a experimental folder as it is not yet fully tested.   

import sys
import os
import json
import time
import numpy as np
import pyigtl
import nibabel as nib
import warnings
import threading
from itertools import combinations
from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor
import pyvista as pv
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=UserWarning)

PROBE_OFFSETS = {
    "Probe_1": np.array([-128.08206577, -11.60711977,  5.77805026]),
    "Probe_2": np.array([-109.69918709,  -4.54499096,  4.47161016]),
}

HOST = "130.89.204.125"
PORT = 18944

# ─────────────────────────────────────────────────────────────────
# THREAD 1 — NDI uitlezen
# ─────────────────────────────────────────────────────────────────
class NDIReaderThread(QtCore.QThread):
    raw_matrix        = QtCore.pyqtSignal(str, np.ndarray)
    connection_status = QtCore.pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self._stop = False

    def run(self):
        try:
            client = pyigtl.OpenIGTLinkClient(self.host, self.port)
            self.connection_status.emit("NDI verbonden (wacht op data...)")
        except Exception as e:
            self.connection_status.emit(f"Verbindingsfout: {e}")
            return

        first_msg = False
        while not self._stop:
            try:
                msgs = client.get_latest_messages()
                for msg in msgs:
                    if isinstance(msg, pyigtl.TransformMessage):
                        if not first_msg:
                            self.connection_status.emit("NDI actief ✓")
                            first_msg = True
                        name = getattr(msg, 'device_name', None) or "unknown"
                        self.raw_matrix.emit(name, msg.matrix.copy().astype(np.float64))
            except Exception:
                if first_msg:
                    self.connection_status.emit("NDI verbinding verloren")
                    first_msg = False
            time.sleep(0.001)

    def stop(self):
        self._stop = True


# ─────────────────────────────────────────────────────────────────
# THREAD 2 — Transformaties berekenen
# ─────────────────────────────────────────────────────────────────
class TransformWorkerThread(QtCore.QThread):
    pose_ready   = QtCore.pyqtSignal(np.ndarray)
    sensor_ready = QtCore.pyqtSignal(np.ndarray, object)

    def __init__(self, probe_offset):
        super().__init__()
        self.probe_offset = probe_offset.copy()
        self._stop        = False
        self._input_lock  = threading.Lock()
        self._pending     = None
        self._reg_lock    = threading.Lock()
        self._R_reg       = None
        self._t_reg       = None
        self._scale       = 1.0

    def push_raw(self, m):
        with self._input_lock:
            self._pending = m

    def set_registration(self, R_reg, t_reg, scale=1.0):
        with self._reg_lock:
            self._R_reg = R_reg.copy()
            self._t_reg = t_reg.copy()
            self._scale = scale

    def clear_registration(self):
        with self._reg_lock:
            self._R_reg = None
            self._t_reg = None
            self._scale = 1.0

    def set_probe_offset(self, offset):
        self.probe_offset = offset.copy()

    def run(self):
        while not self._stop:
            with self._input_lock:
                mat = self._pending
                self._pending = None
            if mat is not None:
                try:
                    euler = R.from_matrix(mat[:3, :3]).as_euler('ZYX', degrees=True)
                except Exception:
                    euler = None
                self.sensor_ready.emit(mat.copy(), euler)

                pos = mat[:3, 3]
                rm  = mat[:3, :3]
                tip = rm @ self.probe_offset + pos

                with self._reg_lock:
                    Rr = self._R_reg
                    tr = self._t_reg
                    sc = self._scale

                if Rr is not None:
                    dt = Rr @ (tip * sc) + tr
                    dr = Rr @ rm
                else:
                    dt, dr = tip, rm

                out = np.eye(4)
                out[:3, :3] = dr
                out[:3,  3] = dt
                self.pose_ready.emit(out)
            else:
                time.sleep(0.0005)

    def stop(self):
        self._stop = True


# ─────────────────────────────────────────────────────────────────
# HOOFD GUI
# ─────────────────────────────────────────────────────────────────
class NDIValidationGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDI Intra-operative Validation")
        self.resize(1520, 880)

        self.probe_type   = "Probe_1"
        self.probe_offset = PROBE_OFFSETS["Probe_1"].copy()

        self.ndi_points         = []
        self.target_points      = []
        self.registration_matrix = None
        self.current_step       = -1
        self.corners_voxels     = []
        self.pointer_actor      = None
        self.highlight_actor    = None
        self.trajectory_actors  = []
        self._reg_idx           = None

        self._display_lock = threading.Lock()
        self._display_mat  = None
        self._last_valid_mat = None
        self._render_dirty = False

        self._build_ui()
        self.load_pipeline_data()
        self._start_threads()

        self._rf = 0
        self._cf = 0
        fps_t = QtCore.QTimer(self)
        fps_t.timeout.connect(self._upd_fps)
        fps_t.start(1000)
        rt = QtCore.QTimer(self)
        rt.timeout.connect(self._render_tick)
        rt.start(8)

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        cw   = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QHBoxLayout(cw)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Links ─────────────────────────────────────────────────
        left = QtWidgets.QWidget()
        left.setFixedWidth(240)
        left.setStyleSheet("background:#1a1a1a;color:white;")
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(6)

        # Probe selectie
        probe_hdr = QtWidgets.QLabel("PROBE")
        probe_hdr.setStyleSheet("color:#4ecdc4;font-size:10px;font-weight:bold;"
                                "border-bottom:1px solid #333;padding-bottom:3px;")
        ll.addWidget(probe_hdr)

        self._probe_combo = QtWidgets.QComboBox()
        self._probe_combo.addItems(["Probe_1", "Probe_2"])
        self._probe_combo.setCurrentText("Probe_1")
        self._probe_combo.setStyleSheet(
            "background:#222;color:white;padding:5px;border:1px solid #444;"
            "font-weight:bold;font-size:12px;")
        self._probe_combo.currentTextChanged.connect(self._on_probe_changed)
        ll.addWidget(self._probe_combo)

        ll.addSpacing(8)

        # Status labels
        self.status_lbl = QtWidgets.QLabel("Status: Initialiseren...")
        self.instr_lbl  = QtWidgets.QLabel("Data laden...")
        self.fps_lbl    = QtWidgets.QLabel("— fps")
        self.dev_lbl    = QtWidgets.QLabel("Device: —")

        for lbl in [self.status_lbl, self.instr_lbl]:
            lbl.setStyleSheet("color:white;font-size:12px;font-weight:bold;")
            lbl.setWordWrap(True)
            ll.addWidget(lbl)
        for lbl in [self.fps_lbl, self.dev_lbl]:
            lbl.setStyleSheet("color:#888;font-size:10px;")
            ll.addWidget(lbl)

        ll.addSpacing(8)

        def mkbtn(txt, bg, fg="white", cb=None):
            b = QtWidgets.QPushButton(txt)
            b.setStyleSheet(
                f"QPushButton{{background:{bg};color:{fg};font-weight:bold;"
                f"padding:7px;border-radius:4px;}}"
                f"QPushButton:disabled{{background:#444;color:#777;}}"
                f"QPushButton:hover{{background:{bg}cc;}}")
            if cb:
                b.clicked.connect(cb)
            return b

        self.next_btn = mkbtn("CAPTURE POINT  (Space)", "#2ecc71", "black",
                              self.on_next_clicked)
        self.next_btn.setEnabled(False)

        recal_btn = mkbtn("↺ Herregistratie", "#e67e22", "white",
                          self._reset_registration)

        for b in [self.next_btn, recal_btn]:
            ll.addWidget(b)

        ll.addStretch()
        root.addWidget(left)

        # ── 3D viewer ─────────────────────────────────────────────
        self.plotter = QtInteractor(cw)
        root.addWidget(self.plotter.interactor, 1)
        # Assen direct instellen zodat ze altijd zichtbaar zijn
        QtCore.QTimer.singleShot(200, lambda: self._add_orientation_axes())

        # ── Rechts ────────────────────────────────────────────────
        right = QtWidgets.QWidget()
        right.setFixedWidth(240)
        right.setStyleSheet("background:#111;color:white;")
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(10, 10, 10, 10)
        rl.setSpacing(4)

        def shdr(t):
            lbl = QtWidgets.QLabel(t)
            lbl.setStyleSheet("color:#4ecdc4;font-size:10px;font-weight:bold;"
                              "border-bottom:1px solid #333;padding-bottom:3px;"
                              "margin-top:8px;")
            return lbl

        # Sensor positie (rauw)
        rl.addWidget(shdr("SENSOR POSITIE (mm, rauw)"))
        self._pos_lbl = {}
        for ax, c in [("X", "#e74c3c"), ("Y", "#2ecc71"), ("Z", "#3498db")]:
            row = QtWidgets.QHBoxLayout()
            la  = QtWidgets.QLabel(ax)
            la.setFixedWidth(16)
            la.setStyleSheet(f"color:{c};font-weight:bold;font-size:12px;")
            lv  = QtWidgets.QLabel("—")
            lv.setStyleSheet("color:white;font-size:13px;font-family:monospace;")
            lv.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(la)
            row.addWidget(lv)
            rl.addLayout(row)
            self._pos_lbl[ax] = lv

        # Sensor oriëntatie (rauw)
        rl.addWidget(shdr("SENSOR ORIËNTATIE (° ZYX, rauw)"))
        self._euler_lbl = {}
        for nm, c in [("Rz yaw", "#e74c3c"), ("Ry pitch", "#2ecc71"), ("Rx roll", "#3498db")]:
            row = QtWidgets.QHBoxLayout()
            la  = QtWidgets.QLabel(nm)
            la.setFixedWidth(72)
            la.setStyleSheet(f"color:{c};font-size:10px;")
            lv  = QtWidgets.QLabel("—")
            lv.setStyleSheet("color:white;font-size:13px;font-family:monospace;")
            lv.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(la)
            row.addWidget(lv)
            rl.addLayout(row)
            self._euler_lbl[nm] = lv

        # Tip positie (na registratie)
        rl.addWidget(shdr("TIP POSITIE (mm, na correctie)"))
        self._tip_lbl = {}
        for ax, c in [("X", "#e74c3c"), ("Y", "#30d274"), ("Z", "#3498db")]:
            row = QtWidgets.QHBoxLayout()
            la  = QtWidgets.QLabel(ax)
            la.setFixedWidth(16)
            la.setStyleSheet(f"color:{c};font-weight:bold;font-size:12px;")
            lv  = QtWidgets.QLabel("—")
            lv.setStyleSheet("color:#f39c12;font-size:13px;font-family:monospace;")
            lv.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            row.addWidget(la)
            row.addWidget(lv)
            rl.addLayout(row)
            self._tip_lbl[ax] = lv

        rl.addStretch()
        root.addWidget(right)

        # Spatiebalk shortcut
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        sc.setContext(QtCore.Qt.ApplicationShortcut)
        sc.activated.connect(self.on_next_clicked)

    # ── Probe wisselen ────────────────────────────────────────────
    def _on_probe_changed(self, name):
        self.probe_type   = name
        self.probe_offset = PROBE_OFFSETS[name].copy()
        self._calc_thread.set_probe_offset(self.probe_offset)
        self._reset_registration()
        self.status_lbl.setText(f"Probe: {name}")

    # ── Herregistratie ────────────────────────────────────────────
    def _reset_registration(self):
        self.ndi_points.clear()
        self.target_points.clear()
        self.registration_matrix = None
        self._calc_thread.clear_registration()
        self.current_step = 0
        self.update_ui_state()
        self.next_btn.show()
        self.next_btn.setEnabled(True)

    # ── Threads ───────────────────────────────────────────────────
    def _start_threads(self):
        self._ndi_thread  = NDIReaderThread(HOST, PORT)
        self._calc_thread = TransformWorkerThread(self.probe_offset)
        self._ndi_thread.connection_status.connect(self.status_lbl.setText)
        self._ndi_thread.raw_matrix.connect(self._on_raw_matrix)
        self._calc_thread.pose_ready.connect(self._on_pose_ready)
        self._calc_thread.sensor_ready.connect(self._on_sensor_ready)
        self._ndi_thread.start()
        self._calc_thread.start()

    # ── Slots ─────────────────────────────────────────────────────
    @QtCore.pyqtSlot(str, np.ndarray)
    def _on_raw_matrix(self, name, mat):
        self.dev_lbl.setText(f"Device: {name}")
        self._calc_thread.push_raw(mat)

    @QtCore.pyqtSlot(np.ndarray, object)
    def _on_sensor_ready(self, mat, euler):
        pos = mat[:3, 3]
        rm  = mat[:3, :3]
        tip = rm @ self.probe_offset + pos
        for ax, v in zip("XYZ", pos):
            self._pos_lbl[ax].setText(f"{v:+9.2f}")
        if euler is not None:
            for k, v in zip(self._euler_lbl.keys(), euler):
                self._euler_lbl[k].setText(f"{v:+8.2f}°")
        else:
            for lv in self._euler_lbl.values():
                lv.setText("invalid")
        for ax, v in zip("XYZ", tip):
            self._tip_lbl[ax].setText(f"{v:+9.2f}")

    @QtCore.pyqtSlot(np.ndarray)
    def _on_pose_ready(self, dm):
        with self._display_lock:
            self._display_mat = dm
            self._last_valid_mat = dm.copy()
        self._render_dirty = True
        self._cf += 1

    def _render_tick(self):
        if not self._render_dirty:
            return
        with self._display_lock:
            mat = self._display_mat
        if mat is None:
            return
        tip = mat[:3, 3]

        if self.pointer_actor is None:
            shaft = pv.Cylinder(center=(0, 0, -80.), direction=(0, 0, 1),
                                radius=2.5, height=160.)
            self.pointer_actor = self.plotter.add_mesh(shaft, color="orange")
        self.pointer_actor.user_matrix = mat

        if self.registration_matrix:
            md = 999.
            for _, actor, target in self.trajectory_actors:
                v  = target - self.entry_voxel
                w  = tip    - self.entry_voxel
                c1, c2 = np.dot(w, v), np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(tip - self.entry_voxel)
                elif c2 <= c1:
                    d = np.linalg.norm(tip - target)
                else:
                    d = np.linalg.norm(tip - (self.entry_voxel + (c1/c2)*v))
                md = min(md, d)
                actor.prop.color      = "green"  if d < 2. else "yellow"
                actor.prop.line_width = 8        if d < 2. else 3
            self.status_lbl.setText(f"Dist to Plan: {md:.1f} mm")
        else:
            self.status_lbl.setText(f"Tip: {tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f} mm")

        self.plotter.render()
        self._render_dirty = False
        self._rf += 1

    def _upd_fps(self):
        self.fps_lbl.setText(f"Render: {self._rf} fps | Calc: {self._cf} fps")
        self._rf = 0
        self._cf = 0

    # ─────────────────────────────────────────────────────────────
    # KUBUS-FIT  —  snap hoekpunten naar perfecte kubus
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def fit_cube_corners(pts, fixed_side=None):
        """
        Snap N gemeten hoekpunten naar een perfecte kubus (alle ribben gelijk).

        Algoritme:
          1. Bereken per as het cluster-gemiddelde laag/hoog  →  3 halfrib-schattingen
          2. Neem het gemiddelde van die 3 schattingen als de éne uniforme halfrib
             (gebaseerd op 3×N/2 metingen i.p.v. N/2 → sqrt(3)x minder ruis)
          3. Snap alle punten naar centroid ± halfrib

        Werkt voor zowel EM-punten als CT-hoekpunten.
        Vereist: punten liggen in twee duidelijke clusters per as.
        """
        pts = np.array(pts, dtype=np.float64)
        centroid = pts.mean(axis=0)
        half_per_ax = []
        for ax in range(3):
            vals = pts[:, ax]
            mid  = (vals.min() + vals.max()) / 2.0
            # Veilige check voor 4 punten tracking als er geen variantie is op een as
            if (vals.max() - vals.min()) < 1e-5:
                continue
            low_mean  = vals[vals <  mid].mean()
            high_mean = vals[vals >= mid].mean()
            half_per_ax.append((high_mean - low_mean) / 2.0)
        
        # Perfecte kubus: één riblengte (gemiddelde, of een vaste waarde)
        if fixed_side is not None:
            half_rib = fixed_side / 2.0
        else:
            if not half_per_ax:
                return pts.copy(), 0.0
            half_rib = float(np.mean(half_per_ax))
            
        snapped = pts.copy()
        for ax in range(3):
            vals = pts[:, ax]
            if (vals.max() - vals.min()) < 1e-5:
                snapped[:, ax] = centroid[ax]
            else:
                mid  = (vals.min() + vals.max()) / 2.0
                snapped[vals <  mid, ax] = centroid[ax] - half_rib
                snapped[vals >= mid, ax] = centroid[ax] + half_rib
        return snapped, half_rib * 2.0   # geeft ook zijlengte terug voor logging

    # ── Registratie helpers ───────────────────────────────────────
    def compute_registration(self):
        P = np.array(self.ndi_points, dtype=np.float64)     # 4×3 NDI-punten (probe-tip)
        Q = np.array(self.target_points, dtype=np.float64)  # 4×3 CT-punten

        if P.shape != (4, 3) or Q.shape != (4, 3):
            self.status_lbl.setText("Registratiefout: verwacht 4 punten in 3D.")
            return

        print("\n=== Geselecteerde punten ===")
        print("NDI punten (probe-tips):")
        for i, p in enumerate(P):
            print(f"  {i+1}: {p}")
        print("Target punten (CT hoeken):")
        for i, q in enumerate(Q):
            print(f"  {i+1}: {q}")

        # ── Kubus-fit (alleen op NDI punten, geforceerd op 50mm) ───────
        P_fit, P_rib = self.fit_cube_corners(P, fixed_side=50.0)
        print(f"\n[Kubus-fit] Zijlengte EM: {P_rib:.2f} mm (Geforceerd)")

        # ── Stap 1: uniforme schaling via alle 6 puntenpaar-afstanden ──
        pairs = list(combinations(range(4), 2))
        dists_P = np.array([np.linalg.norm(P_fit[i] - P_fit[j]) for i, j in pairs], dtype=np.float64)
        dists_Q = np.array([np.linalg.norm(Q[i] - Q[j]) for i, j in pairs], dtype=np.float64)

        if np.any(dists_P < 1e-9):
            self.status_lbl.setText("Registratiefout: dubbele of te dichte NDI-punten.")
            return

        scale = float(np.mean(dists_Q / dists_P))
        P_scaled = P_fit * scale

        # ── Stap 2: volledige 3D-rotatie via Kabsch/SVD/procrustes ───────────────
        cP = np.mean(P_scaled, axis=0)
        cQ = np.mean(Q, axis=0)

        Pc = P_scaled - cP
        Qc = Q - cQ

        H = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T

        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T

        # ── Stap 3: translatie ────────────────────────────────────────
        t_reg = cQ - R_mat @ cP

        # ── Stap 4: FRE ───────────────────────────────────────────────
        P_rot = (R_mat @ P_scaled.T).T
        P_final = P_rot + t_reg
        errors = np.linalg.norm(P_final - Q, axis=1)
        fre = float(np.mean(errors))

        # Euler-hoeken voor logging / GUI
        try:
            euler_zyx = R.from_matrix(R_mat).as_euler("ZYX", degrees=True)
            yaw, pitch, roll = euler_zyx
        except Exception:
            yaw, pitch, roll = np.nan, np.nan, np.nan

        self.registration_matrix = (R_mat, t_reg, scale)
        self._calc_thread.set_registration(R_mat, t_reg, scale)

        msg = (
            f"Registratie OK | "
            f"Rz: {yaw:.1f}° | Ry: {pitch:.1f}° | Rx: {roll:.1f}° | "
            f"Scale: {scale:.4f} | "
            f"FRE: {fre:.2f} mm {'✓' if fre < 2 else '⚠ TE GROOT ↺'}"
        )
        self.status_lbl.setText(msg)

        print("\n=== REGISTRATIE ===")
        print(f"  Schaling:    {scale:.6f}")
        print(f"  Rotatie ZYX: yaw={yaw:.3f}°, pitch={pitch:.3f}°, roll={roll:.3f}°")
        print(f"  R-matrix:\n{R_mat}")
        print(f"  Translatie:  {t_reg}")
        for i, e in enumerate(errors):
            print(f"  Corner {i+1}:  {e:.2f} mm")
        print(f"  FRE:         {fre:.2f} mm")

    def on_next_clicked(self):
        with self._display_lock:
            mat = self._last_valid_mat
        if mat is None:
            self.status_lbl.setText("Fout: Geen NDI data ontvangen!")
            return
        tip = mat[:3, :3] @ self.probe_offset + mat[:3, 3]
        self.ndi_points.append(tip)
        self.target_points.append(self.corners_voxels[self.current_step])
        self.current_step += 1
        if self.current_step == 4:
            self.compute_registration()
        self.update_ui_state()

    # ── Data laden ────────────────────────────────────────────────
    def load_pipeline_data(self):
        sd = os.path.dirname(os.path.abspath(__file__))
        rr = os.path.abspath(os.path.join(sd, "..", ".."))
        jp = os.path.join(rr, "output", "transformed_coords.json")
        ip = os.path.join(rr, "output", "CTpreop_registered.nii.gz")

        if not os.path.exists(jp):
            self.status_lbl.setText("Error: transformed_coords.json niet gevonden!")
            return
        try:
            with open(jp) as f:
                data = json.load(f)

            all_corners      = np.array(data["corners"])
            self.entry_voxel = np.array(data["entry"])
            self.beads_voxels = np.array(data["beads"])

            # Select specific corners by index (3, 4, 7, 8 -> indices 2, 3, 6, 7)
            # This ensures they align perfectly with the white spheres from JSON
            self._reg_idx = np.array([2, 3, 6, 7])
            self.corners_voxels = all_corners[self._reg_idx]
            
            print(f"Registratie corners (3, 4, 7, 8): indices {self._reg_idx+1}")

            self.plotter.set_background("black")

            # CT iso-contour
            if os.path.exists(ip):
                img  = nib.load(ip)
                d    = img.get_fdata()
                mask = (d > 665) & (d < 3850)
                grid = pv.ImageData(dimensions=mask.shape)
                grid.point_data["values"] = mask.flatten(order="F")
                self.plotter.add_mesh(
                    grid.contour([0.5]), color="cyan", opacity=0.1,
                    pickable=False, smooth_shading=False)

            # Beads
            if len(self.beads_voxels) > 0:
                self.plotter.add_mesh(
                    pv.PolyData(self.beads_voxels).glyph(
                        geom=pv.Sphere(), factor=8, orient=False),
                    color="red")

            # Alle corners plotten (wit)
            for pt in all_corners:
                self.plotter.add_mesh(
                    pv.Sphere(center=pt, radius=4),
                    color="white", opacity=0.5)

            # Trajectlijnen — gekleurd en verlengd 30 mm voorbij entry
            TRAJ_COLORS = ['#FF4444', '#44FF44', '#4488FF',
                           '#FF44FF', '#FFAA00', '#00FFFF']
            EXTEND_MM = 95.0
            for idx_t, target in enumerate(self.beads_voxels):
                color = TRAJ_COLORS[idx_t % len(TRAJ_COLORS)]
                # Verleng lijn 30 mm voorbij entry_voxel (buiten de box)
                direction = self.entry_voxel - target
                norm = np.linalg.norm(direction)
                if norm > 0:
                    extended_end = self.entry_voxel + (direction / norm) * EXTEND_MM
                else:
                    extended_end = self.entry_voxel
                line  = pv.Line(target, extended_end)
                actor = self.plotter.add_mesh(line, color=color, line_width=3)
                self.trajectory_actors.append((line, actor, target))

            # XYZ oriëntatie-assen
            self._add_orientation_axes()

            self.current_step = 0
            self.update_ui_state()
            self.plotter.reset_camera()

        except Exception as e:
            print(f"Error loading: {e}")
            self.status_lbl.setText(f"Laad-fout: {e}")

    # ── UI state ──────────────────────────────────────────────────
    def update_ui_state(self):
        if self.current_step < 4:
            pt      = self.corners_voxels[self.current_step]
            self.instr_lbl.setText(
                f"TOUCH CORNER {self.current_step+1}  ({self.current_step+1}/4)\n"
                f"({pt[0]:.0f}, {pt[1]:.0f}, {pt[2]:.0f})")
            self.next_btn.setEnabled(True)
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor = self.plotter.add_mesh(
                pv.Sphere(center=pt, radius=9), color="lime", opacity=0.7)
        else:
            self.instr_lbl.setText("LIVE NAVIGATIE")
            self.next_btn.hide()
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor)
                self.highlight_actor = None


    def _add_orientation_axes(self):
        try:
            origin = [280, 232, 280]  # mm — past bij CT-schaal
            length = 40.0  # mm — past bij CT-schaal

            for vec, label, color in [
                (np.array([1,0,0]), 'X', (1.0, 0.2, 0.2)),
                (np.array([0,1,0]), 'Y', (0.2, 1.0, 0.2)),
                (np.array([0,0,1]), 'Z', (0.3, 0.6, 1.0)),
            ]:
                end = origin + vec * length
                arrow = pv.Arrow(start=origin, direction=vec.astype(float),
                                 tip_length=0.25, tip_radius=0.1,
                                 shaft_radius=0.03, scale=length)
                self.plotter.add_mesh(arrow, color=color, pickable=False)
                # Label iets voorbij de punt  
                label_pos = origin + vec * length * 1.18
                self.plotter.add_point_labels(
                    [label_pos], [label],
                    font_size=16,
                    text_color='white',
                    font_family='arial',
                    bold=True,
                    show_points=False,
                    always_visible=True,
                    shadow=False,
                    pickable=False,
                )
        except Exception as e:
            print(f"Assen fout: {e}")

    def closeEvent(self, event):
        self._ndi_thread.stop()
        self._calc_thread.stop()
        self._ndi_thread.wait(2000)
        self._calc_thread.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w   = NDIValidationGUI()
    w.show()
    sys.exit(app.exec_())
