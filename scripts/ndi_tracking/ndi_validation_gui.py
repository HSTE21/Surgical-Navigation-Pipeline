import sys
import os
import json
import time
import numpy as np
import pyigtl
import nibabel as nib
import warnings
import threading
from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=UserWarning)

PROBE_OFFSETS = {
    "Probe_1": np.array([-128.08206577, -11.60711977, 5.77805026]),
    "Probe_2": np.array([-109.69918709, -4.54499096, 4.47161016])
}
HOST = "130.89.204.125"
PORT = 18944

AXIS_PERMUTATIONS = {
    "XYZ": [0,1,2], "XZY": [0,2,1], "YXZ": [1,0,2],
    "YZX": [1,2,0], "ZXY": [2,0,1], "ZYX": [2,1,0],
}

# ─────────────────────────────────────────────────────────────────
# THREAD 1 — NDI uitlezen
# ─────────────────────────────────────────────────────────────────
class NDIReaderThread(QtCore.QThread):
    raw_matrix        = QtCore.pyqtSignal(str, np.ndarray)
    connection_status = QtCore.pyqtSignal(str)

    def __init__(self, host, port):
        super().__init__()
        self.host  = host; self.port = port; self._stop = False

    def run(self):
        try:
            client = pyigtl.OpenIGTLinkClient(self.host, self.port)
            self.connection_status.emit("NDI verbonden")
        except Exception as e:
            self.connection_status.emit(f"Verbindingsfout: {e}"); return
        while not self._stop:
            try:
                for msg in client.get_latest_messages():
                    if isinstance(msg, pyigtl.TransformMessage):
                        name = getattr(msg, 'device_name', None) or "unknown"
                        self.raw_matrix.emit(name, msg.matrix.copy().astype(np.float64))
            except Exception:
                pass
            time.sleep(0.001)

    def stop(self): self._stop = True


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
        self._R_reg = None; self._t_reg = None
        self._axis_lock = threading.Lock()
        self._perm  = [0,1,2]
        self._signs = np.array([1.,1.,1.])

    def push_raw(self, m):
        with self._input_lock: self._pending = m

    def set_registration(self, R_reg, t_reg):
        with self._reg_lock: self._R_reg=R_reg.copy(); self._t_reg=t_reg.copy()

    def clear_registration(self):
        with self._reg_lock: self._R_reg=None; self._t_reg=None

    def set_axis_correction(self, perm, signs):
        with self._axis_lock: self._perm=perm; self._signs=signs.copy()

    def set_probe_offset(self, offset):
        self.probe_offset = offset.copy()

    def _correct(self, mat4):
        with self._axis_lock: p=list(self._perm); s=self._signs.copy()
        out=mat4.copy()
        out[:3,:3] = mat4[:3,p]*s[np.newaxis,:]
        out[:3, 3] = mat4[p, 3]*s
        return out

    def run(self):
        while not self._stop:
            with self._input_lock: mat=self._pending; self._pending=None
            if mat is not None:
                try: euler=R.from_matrix(mat[:3,:3]).as_euler('ZYX',degrees=True)
                except: euler=None
                self.sensor_ready.emit(mat.copy(), euler)
                mc  = self._correct(mat)
                pos = mc[:3,3]; rm = mc[:3,:3]
                tip = rm @ self.probe_offset + pos
                with self._reg_lock: Rr=self._R_reg; tr=self._t_reg
                if Rr is not None:
                    dt=Rr@tip+tr; dr=Rr@rm
                else:
                    dt,dr=tip,rm
                out=np.eye(4); out[:3,:3]=dr; out[:3,3]=dt
                self.pose_ready.emit(out)
            else: time.sleep(0.0005)

    def stop(self): self._stop=True


# ─────────────────────────────────────────────────────────────────
# PIVOT KALIBRATIE dialoog
# Draai de probe >30 posities terwijl de tip stilstaat
# ─────────────────────────────────────────────────────────────────
class PivotCalibDialog(QtWidgets.QDialog):
    calibration_done = QtCore.pyqtSignal(np.ndarray)   # nieuwe probe offset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pivot / Tip Kalibratie")
        self.setFixedSize(420, 360)
        self.setStyleSheet("background:#1a1a1a;color:white;")
        self._matrices = []   # lijst van 4×4 matrices
        self._collecting = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(8)

        hdr = QtWidgets.QLabel("PIVOT KALIBRATIE")
        hdr.setStyleSheet("color:#4ecdc4;font-size:14px;font-weight:bold;")
        lay.addWidget(hdr)

        info = QtWidgets.QLabel(
            "1.  Zet de probe-tip op een vast punt (bijv. een gaatje in het fantoom).\n"
            "2.  Kantel de probe in minstens 8 richtingen, houd tip STIL.\n"
            "3.  Druk START — de kalibratie verzamelt automatisch 60 posities.\n"
            "4.  Druk BEREKEN wanneer het verzamelen klaar is."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#aaa;font-size:11px;")
        lay.addWidget(info)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0,60); self.progress.setValue(0)
        self.progress.setStyleSheet(
            "QProgressBar{background:#222;border:1px solid #444;border-radius:3px;height:16px;}"
            "QProgressBar::chunk{background:#2ecc71;border-radius:3px;}")
        lay.addWidget(self.progress)

        self.status_lbl = QtWidgets.QLabel("Klaar om te starten")
        self.status_lbl.setStyleSheet("color:#888;font-size:11px;")
        lay.addWidget(self.status_lbl)

        self.result_lbl = QtWidgets.QLabel("")
        self.result_lbl.setStyleSheet("color:#f39c12;font-size:12px;font-family:monospace;")
        lay.addWidget(self.result_lbl)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶  START verzamelen")
        self.start_btn.setStyleSheet(
            "background:#2ecc71;color:black;font-weight:bold;padding:8px;border-radius:4px;")
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)

        self.calc_btn = QtWidgets.QPushButton("⚙  BEREKEN offset")
        self.calc_btn.setEnabled(False)
        self.calc_btn.setStyleSheet(
            "QPushButton{background:#e67e22;color:white;font-weight:bold;padding:8px;border-radius:4px;}"
            "QPushButton:disabled{background:#444;color:#777;}")
        self.calc_btn.clicked.connect(self._compute)
        btn_row.addWidget(self.calc_btn)
        lay.addLayout(btn_row)

        apply_btn = QtWidgets.QPushButton("✔  TOEPASSEN en sluiten")
        apply_btn.setStyleSheet(
            "QPushButton{background:#3498db;color:white;font-weight:bold;padding:8px;border-radius:4px;}"
            "QPushButton:disabled{background:#444;color:#777;}")
        apply_btn.setEnabled(False)
        apply_btn.clicked.connect(self.accept)
        lay.addWidget(apply_btn)
        self._apply_btn = apply_btn

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._collect_tick)
        self._new_offset = None

    def feed_matrix(self, mat: np.ndarray):
        """Aangeroepen vanuit hoofdvenster bij elke nieuwe NDI matrix."""
        if self._collecting:
            self._pending_mat = mat.copy()

    def _start(self):
        self._matrices.clear()
        self._collecting = True
        self._pending_mat = None
        self.progress.setValue(0)
        self.status_lbl.setText("Verzamelen... kantel de probe!")
        self.start_btn.setEnabled(False)
        self._timer.start(100)   # elke 100 ms een sample

    def _collect_tick(self):
        if hasattr(self,'_pending_mat') and self._pending_mat is not None:
            self._matrices.append(self._pending_mat.copy())
            self._pending_mat = None
            n = len(self._matrices)
            self.progress.setValue(n)
            self.status_lbl.setText(f"Verzameld: {n}/60 posities")
            if n >= 60:
                self._timer.stop()
                self._collecting = False
                self.status_lbl.setText("Klaar! Druk BEREKEN.")
                self.calc_btn.setEnabled(True)

    def _compute(self):
        """
        Pivot kalibratie via lineaire methode (Danilchenko & Fitzpatrick 2010).
        Oplost:  R_i * t_tip + p_i = t_pivot  voor alle i
        → samengevoegd als least-squares: A x = b
        """
        mats = self._matrices
        if len(mats) < 8:
            self.status_lbl.setText("Te weinig posities (minimaal 8)")
            return

        # Bouw A en b
        A_rows, b_rows = [], []
        for mat in mats:
            Ri = mat[:3,:3]
            pi = mat[:3, 3]
            A_rows.append(np.hstack([Ri, -np.eye(3)]))   # [R | -I]
            b_rows.append(-pi)

        A = np.vstack(A_rows)   # (3N × 6)
        b = np.concatenate(b_rows)

        # Least-squares oplossing
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        t_tip   = result[:3]   # probe offset in sensor-coordinaten
        t_pivot = result[3:]   # pivot punt in tracker-coordinaten

        # Residuaal = kalibratie nauwkeurigheid
        residuals = A @ result - b
        rms = np.sqrt(np.mean(residuals**2))

        self._new_offset = t_tip
        txt = (f"Offset:  X={t_tip[0]:+.3f}  Y={t_tip[1]:+.3f}  Z={t_tip[2]:+.3f} mm\n"
               f"Pivot:   X={t_pivot[0]:+.1f}  Y={t_pivot[1]:+.1f}  Z={t_pivot[2]:+.1f} mm\n"
               f"RMS residu: {rms:.3f} mm  ({'OK' if rms<1.5 else 'HOOG — meer posities?'})")
        self.result_lbl.setText(txt)
        self._apply_btn.setEnabled(True)
        self.status_lbl.setText("Kalibratie berekend.")

    def get_offset(self):
        return self._new_offset


# ─────────────────────────────────────────────────────────────────
# HOOFD GUI
# ─────────────────────────────────────────────────────────────────
class NDIValidationGUI(QtWidgets.QMainWindow):
    def __init__(self, probe_type="Probe_"):
        super().__init__()
        self.setWindowTitle("NDI Intra-operative Validation")
        self.resize(1520, 880)

        self.probe_type   = probe_type
        self.probe_offset = PROBE_OFFSETS.get(probe_type, np.zeros(3)).copy()

        self.ndi_points=[]
        self.target_points=[]
        self.registration_matrix=None
        self.current_step=-1
        self.corners_voxels=[]
        self.pointer_actor=None
        self.highlight_actor=None
        self.trajectory_actors=[]

        self._display_lock  = threading.Lock()
        self._display_mat   = None
        self._latest_raw    = None          # voor pivot dialoog
        self._render_dirty  = False
        self._cur_perm      = [0,1,2]
        self._cur_signs     = np.array([1.,1.,1.])

        self._build_ui()
        self.load_pipeline_data()
        self._start_threads()

        self._rf=0; self._cf=0
        fps_t=QtCore.QTimer(self); fps_t.timeout.connect(self._upd_fps); fps_t.start(1000)
        rt=QtCore.QTimer(self); rt.timeout.connect(self._render_tick); rt.start(8)

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        cw=QtWidgets.QWidget(); self.setCentralWidget(cw)
        root=QtWidgets.QHBoxLayout(cw); root.setSpacing(0); root.setContentsMargins(0,0,0,0)

        # ── Links ─────────────────────────────────────────────────
        left=QtWidgets.QWidget(); left.setFixedWidth(230)
        left.setStyleSheet("background:#1a1a1a;color:white;")
        ll=QtWidgets.QVBoxLayout(left); ll.setContentsMargins(10,10,10,10); ll.setSpacing(6)

        self.status_lbl = QtWidgets.QLabel("Status: Initialiseren...")
        self.instr_lbl  = QtWidgets.QLabel("Data laden...")
        self.fps_lbl    = QtWidgets.QLabel("— fps")
        self.dev_lbl    = QtWidgets.QLabel("Device: —")

        for l in [self.status_lbl,self.instr_lbl]:
            l.setStyleSheet("color:white;font-size:12px;font-weight:bold;"); ll.addWidget(l)
        for l in [self.fps_lbl,self.dev_lbl]:
            l.setStyleSheet("color:#888;font-size:10px;"); ll.addWidget(l)

        ll.addSpacing(8)

        def mkbtn(txt, bg, fg="white", cb=None):
            b=QtWidgets.QPushButton(txt)
            b.setStyleSheet(f"QPushButton{{background:{bg};color:{fg};font-weight:bold;"
                            f"padding:7px;border-radius:4px;}}"
                            f"QPushButton:disabled{{background:#444;color:#777;}}"
                            f"QPushButton:hover{{background:{bg}cc;}}")
            if cb: b.clicked.connect(cb)
            return b

        self.next_btn   = mkbtn("CAPTURE POINT  (Space)", "#2ecc71", "black", self.on_next_clicked)
        self.next_btn.setEnabled(False)
        recal_btn       = mkbtn("↺  Herregistratie",      "#e67e22", "white", self._reset_registration)
        pivot_btn       = mkbtn("⊕  Pivot Kalibratie",    "#9b59b6", "white", self._open_pivot_calib)

        for b in [self.next_btn, recal_btn, pivot_btn]: ll.addWidget(b)
        ll.addStretch()
        root.addWidget(left)

        # ── 3D ────────────────────────────────────────────────────
        self.plotter=QtInteractor(cw); root.addWidget(self.plotter.interactor,1)

        # ── Rechts ────────────────────────────────────────────────
        right=QtWidgets.QWidget(); right.setFixedWidth(270)
        right.setStyleSheet("background:#111;color:white;")
        rl=QtWidgets.QVBoxLayout(right); rl.setContentsMargins(10,10,10,10); rl.setSpacing(4)

        def shdr(t):
            l=QtWidgets.QLabel(t)
            l.setStyleSheet("color:#4ecdc4;font-size:10px;font-weight:bold;"
                            "border-bottom:1px solid #333;padding-bottom:3px;margin-top:8px;")
            return l

        # Sensor positie
        rl.addWidget(shdr("SENSOR POSITIE  (mm, rauw)"))
        self._pos_lbl={}
        for ax,c in [("X","#e74c3c"),("Y","#2ecc71"),("Z","#3498db")]:
            r=QtWidgets.QHBoxLayout()
            la=QtWidgets.QLabel(ax); la.setFixedWidth(16)
            la.setStyleSheet(f"color:{c};font-weight:bold;font-size:12px;")
            lv=QtWidgets.QLabel("—"); lv.setStyleSheet("color:white;font-size:13px;font-family:monospace;")
            lv.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
            r.addWidget(la); r.addWidget(lv); rl.addLayout(r); self._pos_lbl[ax]=lv

        # Sensor oriëntatie rauw
        rl.addWidget(shdr("SENSOR ORIËNTATIE  (° ZYX, rauw)"))
        self._euler_lbl={}
        for nm,c in [("Rz yaw","#e74c3c"),("Ry pitch","#2ecc71"),("Rx roll","#3498db")]:
            r=QtWidgets.QHBoxLayout()
            la=QtWidgets.QLabel(nm); la.setFixedWidth(72)
            la.setStyleSheet(f"color:{c};font-size:10px;")
            lv=QtWidgets.QLabel("—"); lv.setStyleSheet("color:white;font-size:13px;font-family:monospace;")
            lv.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
            r.addWidget(la); r.addWidget(lv); rl.addLayout(r); self._euler_lbl[nm]=lv

        # Tip positie
        rl.addWidget(shdr("TIP POSITIE  (mm, na correctie)"))
        self._tip_lbl={}
        for ax,c in [("X","#e74c3c"),("Y","#30d274"),("Z","#3498db")]:
            r=QtWidgets.QHBoxLayout()
            la=QtWidgets.QLabel(ax); la.setFixedWidth(16)
            la.setStyleSheet(f"color:{c};font-weight:bold;font-size:12px;")
            lv=QtWidgets.QLabel("—"); lv.setStyleSheet("color:#f39c12;font-size:13px;font-family:monospace;")
            lv.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
            r.addWidget(la); r.addWidget(lv); rl.addLayout(r); self._tip_lbl[ax]=lv

        # Probe offset (live tonen)
        rl.addWidget(shdr("PROBE OFFSET  (mm)"))
        self._offset_lbl=QtWidgets.QLabel(self._fmt_offset())
        self._offset_lbl.setStyleSheet("color:#9b59b6;font-size:11px;font-family:monospace;")
        rl.addWidget(self._offset_lbl)

        # As-permutatie
        rl.addWidget(shdr("AS PERMUTATIE"))
        self._perm_combo=QtWidgets.QComboBox()
        self._perm_combo.addItems(list(AXIS_PERMUTATIONS.keys()))
        self._perm_combo.setCurrentText("XYZ")
        self._perm_combo.setStyleSheet("background:#222;color:white;padding:4px;border:1px solid #444;")
        self._perm_combo.currentTextChanged.connect(self._on_axis_change)
        rl.addWidget(self._perm_combo)

        # Tekenwissel
        rl.addWidget(shdr("TEKENWISSEL"))
        self._sign_btns={}
        sr=QtWidgets.QHBoxLayout()
        for ax in ["X","Y","Z"]:
            b=QtWidgets.QPushButton(f"+{ax}"); b.setCheckable(True); b.setFixedHeight(30)
            b.setStyleSheet("QPushButton{background:#222;color:white;border:1px solid #555;"
                            "border-radius:3px;font-weight:bold;}"
                            "QPushButton:checked{background:#c0392b;border:1px solid #e74c3c;}"
                            "QPushButton:hover{background:#333;}")
            b.clicked.connect(self._on_sign_change)
            sr.addWidget(b); self._sign_btns[ax]=b
        rl.addLayout(sr)

        # Presets
        rl.addWidget(shdr("SNELLE PRESETS"))
        for lbl,perm,signs in [
            ("RAS standaard",  "XYZ", [False,False,False]),
            ("LPS → RAS",      "XYZ", [True, True, False]),
            ("Flip Z",         "XYZ", [False,False,True ]),
            ("ZYX gespiegeld", "ZYX", [True, False,False]),
        ]:
            b=QtWidgets.QPushButton(lbl)
            b.setStyleSheet("QPushButton{background:#1e3a4a;color:#4ecdc4;padding:4px;"
                            "border:1px solid #2c5f7a;border-radius:3px;font-size:10px;}"
                            "QPushButton:hover{background:#2c5f7a;}")
            b.clicked.connect(lambda _,p=perm,s=signs: self._apply_preset(p,s))
            rl.addWidget(b)

        rl.addStretch(); root.addWidget(right)

        sc=QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space),self)
        sc.setContext(QtCore.Qt.ApplicationShortcut); sc.activated.connect(self.on_next_clicked)

    def _fmt_offset(self):
        o=self.probe_offset
        return f"X={o[0]:+.2f}  Y={o[1]:+.2f}  Z={o[2]:+.2f}"

    # ── As-correctie ──────────────────────────────────────────────
    def _on_axis_change(self,t):
        self._cur_perm=AXIS_PERMUTATIONS.get(t,[0,1,2]); self._push_axis()

    def _on_sign_change(self):
        self._cur_signs=np.array([-1. if self._sign_btns[a].isChecked() else 1. for a in "XYZ"])
        self._push_axis()

    def _apply_preset(self,perm_str,sign_bools):
        self._perm_combo.blockSignals(True); self._perm_combo.setCurrentText(perm_str)
        self._perm_combo.blockSignals(False)
        self._cur_perm=AXIS_PERMUTATIONS[perm_str]
        for a,s in zip("XYZ",sign_bools): self._sign_btns[a].setChecked(s)
        self._cur_signs=np.array([-1. if s else 1. for s in sign_bools])
        self._push_axis()

    def _push_axis(self):
        self._calc_thread.set_axis_correction(self._cur_perm,self._cur_signs)

    # ── Pivot kalibratie ──────────────────────────────────────────
    def _open_pivot_calib(self):
        dlg=PivotCalibDialog(self)
        # Koppel NDI data door naar dialoog
        self._pivot_dlg=dlg
        if dlg.exec_()==QtWidgets.QDialog.Accepted:
            new_offset=dlg.get_offset()
            if new_offset is not None:
                self.probe_offset=new_offset
                self._calc_thread.set_probe_offset(new_offset)
                self._offset_lbl.setText(self._fmt_offset())
                self.status_lbl.setText("Pivot kalibratie toegepast ✓")
        self._pivot_dlg=None

    # ── Herregistratie ────────────────────────────────────────────
    def _reset_registration(self):
        self.ndi_points.clear(); self.target_points.clear()
        self.registration_matrix=None
        self._calc_thread.clear_registration()
        self.current_step=0; self.update_ui_state()
        self.status_lbl.setText("Herregistratie gestart")
        self.next_btn.show(); self.next_btn.setEnabled(True)

    # ── Threads ───────────────────────────────────────────────────
    def _start_threads(self):
        self._ndi_thread  = NDIReaderThread(HOST,PORT)
        self._calc_thread = TransformWorkerThread(self.probe_offset)
        self._ndi_thread.connection_status.connect(self.status_lbl.setText)
        self._ndi_thread.raw_matrix.connect(self._on_raw_matrix)
        self._calc_thread.pose_ready.connect(self._on_pose_ready)
        self._calc_thread.sensor_ready.connect(self._on_sensor_ready)
        self._ndi_thread.start(); self._calc_thread.start()

    # ── Slots ─────────────────────────────────────────────────────
    @QtCore.pyqtSlot(str, np.ndarray)
    def _on_raw_matrix(self, name, mat):
        self.dev_lbl.setText(f"Device: {name}")
        self._calc_thread.push_raw(mat)
        if hasattr(self,'_pivot_dlg') and self._pivot_dlg:
            self._pivot_dlg.feed_matrix(mat)

    @QtCore.pyqtSlot(np.ndarray, object)
    def _on_sensor_ready(self, mat, euler):
        pos=mat[:3,3]
        p=self._cur_perm; s=self._cur_signs
        pos_c=mat[p,3]*s
        rm_c=mat[:3,p]*s[np.newaxis,:]
        tip=rm_c@self.probe_offset+pos_c
        for ax,v in zip("XYZ",pos): self._pos_lbl[ax].setText(f"{v:+9.2f}")
        if euler is not None:
            for k,v in zip(self._euler_lbl.keys(),euler): self._euler_lbl[k].setText(f"{v:+8.2f}°")
        else:
            for lv in self._euler_lbl.values(): lv.setText("invalid")
        for ax,v in zip("XYZ",tip): self._tip_lbl[ax].setText(f"{v:+9.2f}")

    @QtCore.pyqtSlot(np.ndarray)
    def _on_pose_ready(self, dm):
        with self._display_lock: self._display_mat=dm
        self._render_dirty=True; self._cf+=1

    def _render_tick(self):
        if not self._render_dirty: return
        with self._display_lock: mat=self._display_mat
        if mat is None: return
        tip=mat[:3,3]
        if self.pointer_actor is None:
            shaft=pv.Cylinder(center=(0,0,-80.),direction=(0,0,1),radius=2.5,height=160.)
            self.pointer_actor=self.plotter.add_mesh(shaft,color="orange")
        self.pointer_actor.user_matrix=mat
        if self.registration_matrix:
            md=999.
            for _,actor,target in self.trajectory_actors:
                v,w=target-self.entry_voxel,tip-self.entry_voxel
                c1,c2=np.dot(w,v),np.dot(v,v)
                if c1<=0: d=np.linalg.norm(tip-self.entry_voxel)
                elif c2<=c1: d=np.linalg.norm(tip-target)
                else: d=np.linalg.norm(tip-(self.entry_voxel+(c1/c2)*v))
                md=min(md,d)
                actor.prop.color="green" if d<2. else "yellow"
                actor.prop.line_width=8 if d<2. else 3
            self.status_lbl.setText(f"Dist to Plan: {md:.1f} mm")
        else:
            self.status_lbl.setText(f"Tip: {tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f} mm")
        self.plotter.render(); self._render_dirty=False; self._rf+=1

    def _upd_fps(self):
        self.fps_lbl.setText(f"Render: {self._rf} fps  |  Calc: {self._cf} fps")
        self._rf=0; self._cf=0

    # ── Registratie: alleen de 4 bovenste corners ─────────────────
    def _select_top4(self, corners: np.ndarray) -> np.ndarray:
        """Geeft indices van de 4 corners met hoogste Z-waarde."""
        idx = np.argsort(corners[:,2])[-4:]   # hoogste Z = bovenste punten
        return idx[np.argsort(idx)]           # behoud originele volgorde

    def compute_registration(self):
        P=np.array(self.ndi_points); Q=np.array(self.target_points)
        cP,cQ=np.mean(P,axis=0),np.mean(Q,axis=0)
        H=(P-cP).T@(Q-cQ)
        U,S,Vt=np.linalg.svd(H); R_mat=Vt.T@U.T
        if np.linalg.det(R_mat)<0: Vt[2,:]*=-1; R_mat=Vt.T@U.T
        t_reg=cQ-R_mat@cP
        self.registration_matrix=(R_mat,t_reg)
        self._calc_thread.set_registration(R_mat,t_reg)
        Pt=(R_mat@P.T).T+t_reg; errors=np.linalg.norm(Pt-Q,axis=1)
        print("\n=== REGISTRATIE (4 bovenste punten) ===")
        for i,e in enumerate(errors): print(f"  Corner {self._top4_idx[i]+1}: {e:.2f} mm")
        fre=np.mean(errors)
        print(f"  FRE: {fre:.2f} mm  ({'OK' if fre<2 else 'TE GROOT'})")
        self.status_lbl.setText(f"FRE: {fre:.2f} mm  ({'OK' if fre<2 else 'TE GROOT ↺'})")

    def on_next_clicked(self):
        with self._display_lock: mat=self._display_mat
        if mat is None: return
        tip=mat[:3,:3]@self.probe_offset+mat[:3,3]
        self.ndi_points.append(tip)
        # Sla de bijhorende target op (al gefilterd op top4)
        self.target_points.append(self.corners_voxels[self._top4_idx[self.current_step]])
        self.current_step+=1
        if self.current_step==4: self.compute_registration()
        self.update_ui_state()

    def load_pipeline_data(self):
        sd=os.path.dirname(os.path.abspath(__file__))
        rr=os.path.abspath(os.path.join(sd,"..",".."))
        jp=os.path.join(rr,"output","transformed_coords.json")
        ip=os.path.join(rr,"output","CTpreop_registered.nii.gz")
        if not os.path.exists(jp):
            self.status_lbl.setText("Error: transformed_coords.json niet gevonden!"); return
        try:
            with open(jp) as f: data=json.load(f)
            all_corners=np.array(data["corners"])
            self.entry_voxel=np.array(data["entry"])
            self.beads_voxels=np.array(data["beads"])

            # ── Selecteer 4 bovenste corners ──────────────────────
            self._top4_idx=self._select_top4(all_corners)
            self.corners_voxels=all_corners[self._top4_idx]
            print(f"Registratie corners (top 4, Z): {self._top4_idx+1} → Z-waarden: {all_corners[self._top4_idx,2].round(1)}")

            self.plotter.set_background("black")
            if os.path.exists(ip):
                img=nib.load(ip); d=img.get_fdata()
                mask=(d>665)&(d<3850)
                grid=pv.ImageData(dimensions=mask.shape)
                grid.point_data["values"]=mask.flatten(order="F")
                self.plotter.add_mesh(grid.contour([0.5]),color="cyan",opacity=0.1,
                                      pickable=False,smooth_shading=False)
            if len(self.beads_voxels)>0:
                self.plotter.add_mesh(
                    pv.PolyData(self.beads_voxels).glyph(geom=pv.Sphere(),factor=8,orient=False),
                    color="red")

            # Alle corners grijs, top-4 wit + groter
            for i,pt in enumerate(all_corners):
                is_top=i in self._top4_idx
                self.plotter.add_mesh(pv.Sphere(center=pt,radius=4 if is_top else 2),
                                      color="white" if is_top else "#555",opacity=0.5)

            for target in self.beads_voxels:
                line=pv.Line(self.entry_voxel,target)
                actor=self.plotter.add_mesh(line,color="yellow",line_width=3)
                self.trajectory_actors.append((line,actor,target))

            self.current_step=0; self.update_ui_state(); self.plotter.reset_camera()
        except Exception as e: print(f"Error loading: {e}")

    def update_ui_state(self):
        if self.current_step < 4:
            pt=self.corners_voxels[self.current_step]
            idx_orig=self._top4_idx[self.current_step]+1
            self.instr_lbl.setText(
                f"TOUCH CORNER {idx_orig}  ({self.current_step+1}/4)\n"
                f"({pt[0]:.0f}, {pt[1]:.0f}, {pt[2]:.0f})")
            self.next_btn.setEnabled(True)
            if self.highlight_actor: self.plotter.remove_actor(self.highlight_actor)
            self.highlight_actor=self.plotter.add_mesh(
                pv.Sphere(center=pt,radius=9),color="lime",opacity=0.7)
        else:
            self.instr_lbl.setText("LIVE NAVIGATIE")
            self.next_btn.hide()
            if self.highlight_actor:
                self.plotter.remove_actor(self.highlight_actor); self.highlight_actor=None

    def closeEvent(self,event):
        self._ndi_thread.stop(); self._calc_thread.stop()
        self._ndi_thread.wait(2000); self._calc_thread.wait(2000)
        super().closeEvent(event)


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    w=NDIValidationGUI(); w.show()
    sys.exit(app.exec_())