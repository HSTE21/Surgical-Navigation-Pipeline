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


# =================================================================
# OFFLINE SIMULATIE  —  zet beide vlaggen True voor standalone test
# USE_SIMULATED_REGISTRATION : slaat 40 captures over, vult direct in
# USE_SIMULATED_NDI_STREAM   : speelt validation track af zonder NDI
# =================================================================
USE_SIMULATED_REGISTRATION = True
USE_SIMULATED_NDI_STREAM   = True


# EM-gemiddelden meting 1..8 (volgorde = corner 1..8 in JSON)
_SIMULATED_EM_8PTS = [[61.249999455043245, 20.555714743477957, -216.96000017438615], [11.15833330154419, 19.71833324432373, -216.66166178385419], [79.43874835968018, 21.740000009536743, -164.3625030517578], [-19.421428680419922, 24.905714307512557, -165.6028551374163], [39.32749938964844, -28.22374963760376, -215.45749855041504], [1.915999984741211, -28.061999893188474, -216.08399963378906], [83.06499989827473, -28.56666660308838, -162.19833119710287], [-48.791250228881836, -35.59499979019165, -154.11000061035156]]


# 40 captures: 5 rondes van 8 corners  (zelfde round-robin als on_next_clicked)
SIMULATED_EM_40PTS = (_SIMULATED_EM_8PTS * 5)


# Validatie track na registratie (timestamp-gestuurd)
SIMULATED_VALIDATION_TRACK = [
    ["2026-06-16 10:24:31.003285","Pointer2ToEmTracker",-47.869998931884766,-37.33000183105469,-154.25999450683594,0],
    ["2026-06-16 10:24:33.024335","Pointer2ToEmTracker",-28.020000457763672,-31.149999618530273,-162.6999969482422,0],
    ["2026-06-16 10:24:33.801018","Pointer2ToEmTracker",32.5,-14.279999732971191,-224.64999389648438,0],
    ["2026-06-16 10:24:33.812452","Pointer2ToEmTracker",29.799999237060547,-13.829999923706055,-218.0399932861328,0],
    ["2026-06-16 10:24:35.823225","Pointer2ToEmTracker",29.079999923706055,-12.069999694824219,-216.69000244140625,0],
    ["2026-06-16 10:24:35.833700","Pointer2ToEmTracker",31.209999084472656,-9.829999923706055,-217.13999938964844,0],
    ["2026-06-16 10:24:41.105042","Pointer2ToEmTracker",28.459999084472656,-9.609999656677246,-213.1199951171875,0],
    ["2026-06-16 10:24:41.116668","Pointer2ToEmTracker",28.450000762939453,-8.850000381469727,-212.25999450683594,0],
    ["2026-06-16 10:24:42.008278","Pointer2ToEmTracker",28.469999313354492,-11.319999694824219,-213.74000549316406,0],
    ["2026-06-16 10:24:42.072185","Pointer2ToEmTracker",29.229999542236328,-10.539999961853027,-214.57000732421875,0],
    ["2026-06-16 10:24:42.624880","Pointer2ToEmTracker",29.93000030517578,-11.029999732971191,-215.6199951171875,0],
    ["2026-06-16 10:24:43.216773","Pointer2ToEmTracker",29.040000915527344,-9.5,-214.02999877929688,0],
    ["2026-06-16 10:24:44.041159","Pointer2ToEmTracker",29.670000076293945,-8.680000305175781,-214.19000244140625,0],
    ["2026-06-16 10:24:44.065126","Pointer2ToEmTracker",29.6200008392334,-8.630000114440918,-214.3300018310547,0],
    ["2026-06-16 10:24:44.136450","Pointer2ToEmTracker",29.649999618530273,-8.710000038146973,-214.47000122070312,0],
    ["2026-06-16 10:24:44.717884","Pointer2ToEmTracker",30.15999984741211,-9.1899995803833,-215.3000030517578,0],
    ["2026-06-16 10:24:45.349134","Pointer2ToEmTracker",30.360000610351562,-2.5899999141693115,-212.55999755859375,0],
    ["2026-06-16 10:24:46.833775","Pointer2ToEmTracker",28.75,-8.8100004196167,-211.9499969482422,0],
    ["2026-06-16 10:24:47.871358","Pointer2ToEmTracker",29.200000762939453,-9.100000381469727,-213.80999755859375,0],
    ["2026-06-16 10:24:48.621579","Pointer2ToEmTracker",29.829999923706055,-6.909999847412109,-213.38999938964844,0],
    ["2026-06-16 10:24:49.517691","Pointer2ToEmTracker",36.2599983215332,-7.989999771118164,-214.52000427246094,0]
]




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



    @staticmethod
    def _xyz_to_matrix(x, y, z):
        m = np.eye(4, dtype=np.float64)
        m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
        return m


    def _run_simulation(self):
        """Speelt SIMULATED_VALIDATION_TRACK af op basis van originele timestamps."""
        self.connection_status.emit("Simulatie actief ✓ (offline mode)")
        from datetime import datetime
        prev_ts = None
        time_scale = 0.4
        for row in SIMULATED_VALIDATION_TRACK:
            if self._stop:
                break
            ts_str, device, x, y, z, _ev = row
            if prev_ts is not None:
                try:
                    dt = (datetime.fromisoformat(ts_str) -
                          datetime.fromisoformat(prev_ts)).total_seconds()
                    time.sleep(max(dt * time_scale, 0.01))
                except Exception:
                    time.sleep(0.05)
            prev_ts = ts_str
            self.raw_matrix.emit(device, self._xyz_to_matrix(x, y, z))
        self.connection_status.emit("Simulatie klaar — track afgespeeld")


    def run(self):
        if USE_SIMULATED_NDI_STREAM:
            self._run_simulation()
            return
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
        self.total_capture_count = 0
        self.repeats            = 5
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



        self.plotter = QtInteractor(cw)
        root.addWidget(self.plotter.interactor, 1)
        QtCore.QTimer.singleShot(200, lambda: self._add_orientation_axes())



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



        sc = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        sc.setContext(QtCore.Qt.ApplicationShortcut)
        sc.activated.connect(self.on_next_clicked)



    def _on_probe_changed(self, name):
        self.probe_type   = name
        self.probe_offset = PROBE_OFFSETS[name].copy()
        self._calc_thread.set_probe_offset(self.probe_offset)
        self._reset_registration()
        self.status_lbl.setText(f"Probe: {name}")



    def _reset_registration(self):
        self.ndi_points.clear()
        self.target_points.clear()
        self.registration_matrix = None
        self._calc_thread.clear_registration()
        self.current_step = 0
        self.update_ui_state()
        self.next_btn.show()
        self.next_btn.setEnabled(True)



    def _start_threads(self):
        self._ndi_thread  = NDIReaderThread(HOST, PORT)
        self._calc_thread = TransformWorkerThread(self.probe_offset)
        self._ndi_thread.connection_status.connect(self.status_lbl.setText)
        self._ndi_thread.raw_matrix.connect(self._on_raw_matrix)
        self._calc_thread.pose_ready.connect(self._on_pose_ready)
        self._calc_thread.sensor_ready.connect(self._on_sensor_ready)
        self._ndi_thread.start()
        self._calc_thread.start()



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
    # KUBUS-FIT  —  snap 8 hoekpunten naar perfecte kubus
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def fit_cube_corners(pts):
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
            low_mean  = vals[vals <  mid].mean()
            high_mean = vals[vals >= mid].mean()
            half_per_ax.append((high_mean - low_mean) / 2.0)
        # Perfecte kubus: één riblengte als gemiddelde over alle 3 assen
        half_rib = float(np.mean(half_per_ax))
        snapped = pts.copy()
        for ax in range(3):
            vals = pts[:, ax]
            mid  = (vals.min() + vals.max()) / 2.0
            snapped[vals <  mid, ax] = centroid[ax] - half_rib
            snapped[vals >= mid, ax] = centroid[ax] + half_rib
        return snapped, half_rib * 2.0   # geeft ook zijlengte terug voor logging



    def compute_registration(self):
        P = np.array(self.ndi_points, dtype=np.float64)
        Q = np.array(self.target_points, dtype=np.float64)



        if P.ndim != 2 or Q.ndim != 2 or P.shape != Q.shape or P.shape[1] != 3 or P.shape[0] < 3:
            self.status_lbl.setText("Registratiefout: verwacht minimaal 3 corresponderende punten in 3D.")
            return



        print("\n=== DEBUG: Geselecteerde punten ===")
        print(f"NDI punten (probe-tips): {P.shape[0]} stuks")
        for i, p in enumerate(P):
            print(f"  {i+1}: {p}")
        print(f"Target punten (CT hoeken): {Q.shape[0]} stuks")
        for i, q in enumerate(Q):
            print(f"  {i+1}: {q}")



        # ── Kubus-fit: forceer gelijke riblengte over alle 3 assen ──────
        # Met 5 herhalingen per hoek en 8 hoeken middelt meetruis (±0.5 mm)
        # weg naar ±0.5/sqrt(5×4) ≈ 0.11 mm per riblengte-schatting.
        # Doordat we het gemiddelde nemen over 3 assen daalt dit verder
        # naar ±0.11/sqrt(3) ≈ 0.06 mm voor de definitieve halfrib.
        P_fit, P_rib = self.fit_cube_corners(P)
        Q_fit, Q_rib = self.fit_cube_corners(Q)
        print(f"\n[Kubus-fit] Zijlengte EM: {P_rib:.2f} mm | CT: {Q_rib:.2f} mm")
        print(f"[Kubus-fit] Verschil:     {abs(P_rib - Q_rib):.2f} mm  "
              f"({'OK' if abs(P_rib - Q_rib) < 5 else 'GROOT — check metingen'})")



        scale = 4.0
        P_scaled = P_fit * scale



        cP = np.mean(P_scaled, axis=0)
        cQ = np.mean(Q_fit, axis=0)



        Pc = P_scaled - cP
        Qc = Q_fit - cQ



        H = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T



        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T



        t_reg = cQ - R_mat @ cP



        P_rot = (R_mat @ P_scaled.T).T
        P_final = P_rot + t_reg
        errors = np.linalg.norm(P_final - Q_fit, axis=1)
        fre = float(np.mean(errors))



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




    # ── Offline simulatie registratie (8 punten x 5 rondes) ─────
    def _apply_simulated_registration(self):
        """
        Vult ndi_points en target_points met 40 hardcoded EM-punten
        (8 metingen x 5 rondes) — exact dezelfde round-robin als on_next_clicked.
        """
        n_corners = len(self.corners_voxels)
        n_total   = self.total_capture_count  # = n_corners * repeats
        if n_corners == 0 or n_total == 0:
            self.status_lbl.setText("Simulatie fout: corners niet geladen.")
            return


        em_pts = SIMULATED_EM_40PTS   # altijd genoeg; we slicen op n_total
        self.ndi_points    = [np.asarray(em_pts[i], dtype=np.float64)
                              for i in range(n_total)]
        self.target_points = [np.asarray(self.corners_voxels[i % n_corners], dtype=np.float64)
                              for i in range(n_total)]
        self.current_step  = n_total


        print("\n=== SIMULATIE REGISTRATIE (8 corners × 5 rondes) ===")
        for i in range(n_corners):
            em_mean_i = np.mean([self.ndi_points[i + r * n_corners]
                                 for r in range(self.repeats)], axis=0)
            print(f"  Corner {i+1} | EM gem={np.round(em_mean_i,2)} -> CT={np.round(self.corners_voxels[i],2)}")


        self.compute_registration()
        self.update_ui_state()


    def on_next_clicked(self):
        with self._display_lock:
            mat = self._last_valid_mat
        if mat is None:
            self.status_lbl.setText("Fout: Geen NDI data ontvangen!")
            return
        tip = mat[:3, :3] @ self.probe_offset + mat[:3, 3]
        target_idx = self.current_step % len(self.corners_voxels)
        self.ndi_points.append(tip)
        self.target_points.append(self.corners_voxels[target_idx])
        self.current_step += 1
        if self.current_step == self.total_capture_count:
            self.compute_registration()
        self.update_ui_state()



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



            self._reg_idx = np.arange(len(all_corners))
            self.corners_voxels = all_corners.copy()
            self.total_capture_count = len(self.corners_voxels) * self.repeats
            print(f"Registratie corners: alle {len(self.corners_voxels)} punten gebruikt, {self.repeats} rondes = {self.total_capture_count} captures")



            self.plotter.set_background("black")



            if os.path.exists(ip):
                img  = nib.load(ip)
                d    = img.get_fdata()
                mask = (d > 665) & (d < 3850)
                grid = pv.ImageData(dimensions=mask.shape)
                grid.point_data["values"] = mask.flatten(order="F")
                self.plotter.add_mesh(
                    grid.contour([0.5]), color="cyan", opacity=0.1,
                    pickable=False, smooth_shading=False)



            if len(self.beads_voxels) > 0:
                self.plotter.add_mesh(
                    pv.PolyData(self.beads_voxels).glyph(
                        geom=pv.Sphere(), factor=8, orient=False),
                    color="red")



            for pt in all_corners:
                self.plotter.add_mesh(
                    pv.Sphere(center=pt, radius=4),
                    color="white", opacity=0.5)



            TRAJ_COLORS = ['#FF4444', '#44FF44', '#4488FF',
                           '#FF44FF', '#FFAA00', '#00FFFF']
            EXTEND_MM = 95.0
            for idx_t, target in enumerate(self.beads_voxels):
                color = TRAJ_COLORS[idx_t % len(TRAJ_COLORS)]
                direction = self.entry_voxel - target
                norm = np.linalg.norm(direction)
                if norm > 0:
                    extended_end = self.entry_voxel + (direction / norm) * EXTEND_MM
                else:
                    extended_end = self.entry_voxel
                line  = pv.Line(target, extended_end)
                actor = self.plotter.add_mesh(line, color=color, line_width=3)
                self.trajectory_actors.append((line, actor, target))



            self._add_orientation_axes()



            self.current_step = 0
            self.update_ui_state()
            self.plotter.reset_camera()
            if USE_SIMULATED_REGISTRATION:
                QtCore.QTimer.singleShot(300, self._apply_simulated_registration)



        except Exception as e:
            print(f"Error loading: {e}")
            self.status_lbl.setText(f"Laad-fout: {e}")



    def update_ui_state(self):
        if self.current_step < self.total_capture_count:
            target_idx = self.current_step % len(self.corners_voxels)
            round_idx  = (self.current_step // len(self.corners_voxels)) + 1
            pt         = self.corners_voxels[target_idx]
            self.instr_lbl.setText(
                f"TOUCH CORNER {target_idx+1}  ({self.current_step+1}/{self.total_capture_count})\n"
                f"Ronde {round_idx}/{self.repeats}\n"
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
            self.next_btn.hide()
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor)
                self.highlight_actor = None



    def _add_orientation_axes(self):
        try:
            origin = [280, 232, 280]
            length = 40.0



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
                label_pos = origin + vec * length * 1.18
                self.plotter.add_point_labels(
                    [label_pos], [label],
                    font_size=16, text_color='white', font_family='arial',
                    bold=True, show_points=False, always_visible=True,
                    shadow=False, pickable=False,
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