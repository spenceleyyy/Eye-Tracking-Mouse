"""
tobii_gaze_mouse.py
───────────────────
CALIBRATION FLOW (default — smooth pursuit):
  Phase 1 — Tobii screen calibration (5 points, standard)
  Phase 2 — Smooth pursuit: follow a moving dot (figure-8, ~15 s)
             Collects hundreds of gaze→screen pairs across the full screen
  Phase 3 — Validation grid: 9-point accuracy check with error display
             Accept or redo if accuracy looks poor

CALIBRATION FLOW (legacy — edge trace, --legacy flag):
  Phase 1 — Tobii screen calibration (5 points, standard)
  Phase 2 — Edge trace (20 points around the border)

LIVE CONTROLS:
  Eyes move      →  mouse follows gaze (corrected)
  Gaze + hold    →  cursor snaps & locks to dwell centroid
  Double-blink   →  left-click at locked position
  Esc / Ctrl+C   →  quit
  Mouse top-left →  emergency kill (pyautogui failsafe)

Requirements:
    py -3.10 -m pip install tobii_research pyautogui pywinauto numpy
Run:
    py -3.10 tobii_gaze_mouse.py
    py -3.10 tobii_gaze_mouse.py --legacy   (use old edge-trace calibration)
"""

import sys, time, threading, collections, math, tkinter as tk, argparse

try:
    import tobii_research as tr
except ImportError:
    print("ERROR: tobii_research not installed.\n"
          "  py -3.10 -m pip install tobii_research pyautogui pywinauto")
    sys.exit(1)

try:
    import pyautogui
except ImportError:
    print("ERROR: pyautogui not installed.\n  py -3.10 -m pip install pyautogui")
    sys.exit(1)

try:
    import pywinauto
    HAS_PYWINAUTO = True
except ImportError:
    HAS_PYWINAUTO = False
    print("WARNING: pywinauto not installed — UI gravity disabled.\n"
          "  py -3.10 -m pip install pywinauto")

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

# ─── TUNING ───────────────────────────────────────────────────────────────────
BLINK_MIN_MS           = 80
BLINK_MAX_MS           = 600
DOUBLE_BLINK_WINDOW_MS = 500
COOLDOWN_MS            = 800

GAZE_Y_OFFSET          = 0.0    # manual Y bias — set to 0 when polynomial correction is active

# Smooth pursuit calibration settings
PURSUIT_LAG_MS         = 120    # ms — how far behind the dot your eyes trail during pursuit
                                 # increase if validation errors point consistently ahead of targets

GAZE_SMOOTH_SAMPLES    = 20
MOUSE_SMOOTHING        = 0.75  # higher = smoother/slower, lower = snappier
MOVE_THRESHOLD_PX      = 10

DWELL_TIME_MS          = 300
DWELL_RADIUS_PX        = 60

# Edge gravity — disabled
EDGE_GRAVITY_ZONE      = 0.0
EDGE_GRAVITY_STRENGTH  = 0.0

# UI element gravity — disabled
UI_GRAVITY_RADIUS_PX   = 0
UI_GRAVITY_STRENGTH    = 0.0
UI_SCAN_INTERVAL_S     = 3.0

# Precision mode: when gaze is stable, cursor slows to a crawl for fine control
PRECISION_STABLE_MS    = 400    # how long gaze must be still to enter precision mode
PRECISION_STABLE_PX    = 40     # radius within which gaze counts as "still"
PRECISION_SMOOTHING    = 0.92   # smoothing in precision mode (much higher = much slower)
# ──────────────────────────────────────────────────────────────────────────────

DOT_RADIUS = 20
DOT_INNER  = 6
DWELL_MS   = 2200   # longer dwell per calibration point = more samples = better fit
SHRINK_MS  = 500

# Standard 5-point Tobii calibration
TOBII_CALIB_POINTS = [
    (0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
]

# 20-point edge trace: corners + thirds along each edge, clockwise
# These are the TRUE screen positions we show the user
EDGE_TRACE_POINTS_NORM = [
    # Top edge L→R
    (0.0, 0.0), (0.33, 0.0), (0.67, 0.0), (1.0, 0.0),
    # Right edge T→B
    (1.0, 0.0), (1.0, 0.33), (1.0, 0.67), (1.0, 1.0),
    # Bottom edge R→L
    (1.0, 1.0), (0.67, 1.0), (0.33, 1.0), (0.0, 1.0),
    # Left edge B→T
    (0.0, 1.0), (0.0, 0.67), (0.0, 0.33), (0.0, 0.0),
    # Extra centre-edge midpoints for better interior coverage
    (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5),
]


# ─── Edge Correction Model ────────────────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not found — using fallback 1D correction (less accurate).\n"
          "  Install with:  py -3.10 -m pip install numpy")


class EdgeCorrectionModel:
    """
    Fits a 2D polynomial warp from observed gaze → true screen coords.

    With numpy: fits degree-2 polynomial surfaces for X and Y independently
    using all calibration points together, capturing the nonlinear vertical
    drift that is worst at screen edges.

    Without numpy: falls back to per-axis piecewise linear interpolation.
    """

    def __init__(self):
        # For numpy polynomial fit
        self._cx     = None   # coefficients for true_x = poly2d(gaze_x, gaze_y)
        self._cy     = None   # coefficients for true_y = poly2d(gaze_x, gaze_y)
        # For fallback
        self.x_pairs = []
        self.y_pairs = []
        self._ready  = False
        self._samples = []    # [(gaze_nx, gaze_ny, true_nx, true_ny), ...]

    def add_sample(self, true_nx, true_ny, gaze_nx, gaze_ny):
        self._samples.append((gaze_nx, gaze_ny, true_nx, true_ny))
        self.x_pairs.append((gaze_nx, true_nx))
        self.y_pairs.append((gaze_ny, true_ny))

    def fit(self):
        if HAS_NUMPY and len(self._samples) >= 6:
            self._fit_poly()
        else:
            self.x_pairs.sort(key=lambda p: p[0])
            self.y_pairs.sort(key=lambda p: p[0])
        self._ready = len(self._samples) >= 4

    def _fit_poly(self):
        """Fit affine (degree-1) transform: true = a + b*gx + c*gy
        Degree-1 is more robust than degree-2 — less overfitting risk."""
        gxs = np.array([s[0] for s in self._samples])
        gys = np.array([s[1] for s in self._samples])
        txs = np.array([s[2] for s in self._samples])
        tys = np.array([s[3] for s in self._samples])

        # Build design matrix for affine (degree-1) transform
        A = np.column_stack([
            np.ones_like(gxs),
            gxs, gys,
        ])

        # Least-squares fit for each output axis
        cx, _, _, _ = np.linalg.lstsq(A, txs, rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, tys, rcond=None)

        # Check residuals — if fit is poor, don't use it
        pred_x = A @ cx
        pred_y = A @ cy
        rmse_x = np.sqrt(np.mean((pred_x - txs)**2))
        rmse_y = np.sqrt(np.mean((pred_y - tys)**2))
        print(f"  Calibration fit RMSE — X: {rmse_x:.4f}  Y: {rmse_y:.4f}  "
              f"(lower is better; >0.08 suggests noisy calibration)")

        # Only use polynomial if fit quality is acceptable
        MAX_RMSE = 0.12
        if rmse_x < MAX_RMSE and rmse_y < MAX_RMSE:
            self._cx = cx
            self._cy = cy
            print("  ✓ 2D polynomial correction active.")
        else:
            print("  ⚠ Fit quality poor — falling back to 1D correction. "
                  "Try redoing the edge calibration, staring very precisely at each dot.")
            self._cx = None
            self._cy = None

    def _poly_eval(self, coeffs, gx, gy):
        return coeffs[0] + coeffs[1]*gx + coeffs[2]*gy

    def correct(self, nx, ny):
        """Remap raw normalised gaze → corrected normalised gaze."""
        if not self._ready:
            return nx, ny
        if HAS_NUMPY and self._cx is not None:
            return (self._poly_eval(self._cx, nx, ny),
                    self._poly_eval(self._cy, nx, ny))
        # Fallback: 1D piecewise linear
        return (self._interp(nx, self.x_pairs),
                self._interp(ny, self.y_pairs))

    @staticmethod
    def _interp(v, pairs):
        if not pairs:
            return v
        if v <= pairs[0][0]:
            return pairs[0][1]
        if v >= pairs[-1][0]:
            return pairs[-1][1]
        for i in range(len(pairs) - 1):
            x0, y0 = pairs[i]
            x1, y1 = pairs[i + 1]
            if x0 <= v <= x1:
                t = (v - x0) / (x1 - x0) if x1 != x0 else 0
                return y0 + t * (y1 - y0)
        return v


# ─── UI Element Scanner ───────────────────────────────────────────────────────

class UIScanner:
    """
    Uses pywinauto / Windows UIAutomation to find interactive elements
    on screen and returns their bounding rects as gravity targets.
    Runs in a background thread on a timer so it never blocks gaze.
    """

    # Control types we care about (pywinauto auto_id / control_type strings)
    WANTED_TYPES = {
        "Button", "MenuItem", "TabItem", "Hyperlink",
        "Edit", "ComboBox", "ListItem", "TreeItem",
        "CheckBox", "RadioButton", "Slider", "ToolBar",
    }

    def __init__(self, sw, sh):
        self.sw = sw
        self.sh = sh
        self.targets   = []   # [(cx, cy, label), ...]
        self._lock     = threading.Lock()
        self._running  = True
        self._thread   = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

    def get_targets(self):
        with self._lock:
            return list(self.targets)

    def stop(self):
        self._running = False

    def _scan_loop(self):
        while self._running:
            try:
                self._scan()
            except Exception:
                pass
            time.sleep(UI_SCAN_INTERVAL_S)

    def _scan(self):
        if not HAS_PYWINAUTO:
            return

        from pywinauto import Desktop
        targets = []

        try:
            desktop = Desktop(backend="uia")
            windows = desktop.windows()
        except Exception:
            return

        for win in windows:
            try:
                if not win.is_visible():
                    continue
                wr = win.rectangle()
                # Skip windows completely off-screen
                if wr.right < 0 or wr.bottom < 0:
                    continue
                if wr.left > self.sw or wr.top > self.sh:
                    continue

                # Walk descendants for interactive elements
                try:
                    descendants = win.descendants(control_type=None)
                except Exception:
                    continue

                for el in descendants:
                    try:
                        ct = el.element_info.control_type
                        if ct not in self.WANTED_TYPES:
                            continue
                        if not el.is_visible():
                            continue
                        r = el.rectangle()
                        cx = (r.left + r.right)  // 2
                        cy = (r.top  + r.bottom) // 2
                        # Keep only on-screen elements
                        if 0 <= cx <= self.sw and 0 <= cy <= self.sh:
                            label = ""
                            try:
                                label = el.window_text()[:30]
                            except Exception:
                                pass
                            targets.append((cx, cy, label))
                    except Exception:
                        continue
            except Exception:
                continue

        with self._lock:
            self.targets = targets


# ─── Legacy Calibration Window (edge trace) ───────────────────────────────────

class LegacyCalibrationWindow:
    """
    Two-phase calibration (original / backup):
      Phase 1: standard Tobii 5-point calibration
      Phase 2: 20-point edge trace to build EdgeCorrectionModel
    Returns (accepted: bool, model: EdgeCorrectionModel)
    Run with:  py -3.10 tobii_gaze_mouse.py --legacy
    """

    def __init__(self, tracker):
        self.tracker    = tracker
        self.accepted   = False
        self.edge_model = EdgeCorrectionModel()

        self.root = tk.Tk()
        self.root.title("Tobii Calibration")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost",    True)

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Gaze data collection for edge calibration
        self._gaze_samples    = []
        self._collecting      = False
        self._gaze_lock       = threading.Lock()

        self.root.bind("<Escape>", lambda e: self._abort())
        self._draw_intro()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _clear_widgets(self):
        for w in self.root.winfo_children():
            if isinstance(w, tk.Button):
                w.destroy()

    def _cx(self): return self.sw // 2
    def _cy(self): return self.sh // 2

    # ── Phase 0: intro ────────────────────────────────────────────────────────

    def _draw_intro(self):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        self.canvas.create_text(cx, cy - 120,
            text="Tobii Gaze Mouse — Calibration",
            fill="white", font=("Helvetica", 32, "bold"))
        self.canvas.create_text(cx, cy - 40,
            text="This has two phases:\n\n"
                 "  Phase 1 — Standard 5-point Tobii calibration\n"
                 "  Phase 2 — Edge trace: follow 20 dots around the screen border\n\n"
                 "Phase 2 measures where YOUR eyes actually look when targeting edges,\n"
                 "so the cursor can reach every corner and edge precisely.",
            fill="#aaaaaa", font=("Helvetica", 16), justify="center")
        self.canvas.create_text(cx, cy + 110,
            text="Click anywhere to begin Phase 1",
            fill="#55aaff", font=("Helvetica", 20))
        self.canvas.bind("<Button-1>", lambda e: self._start_tobii_calib())

    # ── Phase 1: Tobii calibration ────────────────────────────────────────────

    def _start_tobii_calib(self):
        self.canvas.unbind("<Button-1>")
        calib = tr.ScreenBasedCalibration(self.tracker)
        calib.enter_calibration_mode()
        threading.Thread(target=self._run_tobii_points,
                         args=(calib,), daemon=True).start()

    def _run_tobii_points(self, calib):
        total = len(TOBII_CALIB_POINTS)
        for i, (nx, ny) in enumerate(TOBII_CALIB_POINTS):
            px, py = int(nx * self.sw), int(ny * self.sh)
            self.root.after(0, self._draw_dot, px, py, DOT_RADIUS,
                            f"Phase 1 of 2 — Point {i+1} of {total}")
            time.sleep(SHRINK_MS / 1000)
            calib.collect_data(nx, ny)
            time.sleep(DWELL_MS / 1000)
        result = calib.compute_and_apply()
        calib.leave_calibration_mode()
        self.root.after(0, self._show_tobii_result, result)

    def _show_tobii_result(self, result):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()

        if result.status == tr.CALIBRATION_STATUS_SUCCESS:
            self.canvas.create_text(cx, cy - 80,
                text="Phase 1 complete  ✓",
                fill="#55ff88", font=("Helvetica", 28, "bold"))
            self.canvas.create_text(cx, cy - 20,
                text="Tobii calibration successful.\n"
                     "Now we'll trace the screen edges to personalise your edge mapping.",
                fill="#aaaaaa", font=("Helvetica", 16), justify="center")
            btn = tk.Button(self.root, text="Continue to Edge Trace →",
                bg="#1a4e8c", fg="white", font=("Helvetica", 18, "bold"),
                padx=20, pady=10, relief="flat", cursor="hand2",
                command=self._start_edge_intro)
            self.canvas.create_window(cx, cy + 70, window=btn)
        else:
            self.canvas.create_text(cx, cy - 80,
                text="Phase 1 failed — please retry",
                fill="#ff5555", font=("Helvetica", 28, "bold"))
            btn = tk.Button(self.root, text="↺  Retry Phase 1",
                bg="#2a2a2a", fg="white", font=("Helvetica", 18),
                padx=20, pady=10, relief="flat", cursor="hand2",
                command=self._draw_intro)
            self.canvas.create_window(cx, cy + 70, window=btn)

    # ── Phase 2: Edge trace ───────────────────────────────────────────────────

    def _start_edge_intro(self):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        self.canvas.create_text(cx, cy - 80,
            text="Phase 2 — Edge Trace",
            fill="white", font=("Helvetica", 28, "bold"))
        self.canvas.create_text(cx, cy - 10,
            text="20 dots will appear around the screen border.\n"
                 "Look directly at each dot as it appears.\n"
                 "The dot shrinks as we collect your gaze — hold still until it moves.\n\n"
                 "This calibrates the edge correction to your eyes specifically.",
            fill="#aaaaaa", font=("Helvetica", 16), justify="center")
        self.canvas.create_text(cx, cy + 100,
            text="Click anywhere to begin",
            fill="#55aaff", font=("Helvetica", 20))
        self.canvas.bind("<Button-1>", lambda e: self._start_edge_trace())

    def _start_edge_trace(self):
        self.canvas.unbind("<Button-1>")
        # Subscribe to gaze for collection
        self.tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self._collect_gaze, as_dictionary=True)
        threading.Thread(target=self._run_edge_points, daemon=True).start()

    def _collect_gaze(self, gaze_data):
        if not self._collecting:
            return
        l_pt = gaze_data["left_gaze_point_on_display_area"]
        r_pt = gaze_data["right_gaze_point_on_display_area"]
        lv   = gaze_data["left_gaze_point_validity"]
        rv   = gaze_data["right_gaze_point_validity"]
        pts  = []
        if lv and l_pt[0] == l_pt[0]: pts.append(l_pt)
        if rv and r_pt[0] == r_pt[0]: pts.append(r_pt)
        if pts:
            ax = sum(p[0] for p in pts) / len(pts)
            ay = sum(p[1] for p in pts) / len(pts)
            with self._gaze_lock:
                self._gaze_samples.append((ax, ay))

    def _run_edge_points(self):
        total = len(EDGE_TRACE_POINTS_NORM)
        for i, (nx, ny) in enumerate(EDGE_TRACE_POINTS_NORM):
            # Inset slightly from hard pixel edge so dot is visible
            margin = 30
            px = int(nx * (self.sw - 2*margin)) + margin
            py = int(ny * (self.sh - 2*margin)) + margin

            self.root.after(0, self._draw_dot, px, py, DOT_RADIUS,
                            f"Phase 2 of 2 — Edge point {i+1} of {total}",
                            "#ff9933")

            # Brief pause before collecting so eyes can land on dot
            time.sleep(SHRINK_MS / 1000)

            # Collect gaze for this point
            with self._gaze_lock:
                self._gaze_samples = []
            self._collecting = True
            time.sleep(DWELL_MS / 1000)
            self._collecting = False

            # Average samples → this person's gaze when looking at (nx, ny)
            with self._gaze_lock:
                samples = list(self._gaze_samples)

            if samples:
                avg_gx = sum(s[0] for s in samples) / len(samples)
                avg_gy = sum(s[1] for s in samples) / len(samples)
                self.edge_model.add_sample(nx, ny, avg_gx, avg_gy)

        self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._collect_gaze)
        self.edge_model.fit()
        self.root.after(0, self._show_edge_result)

    def _show_edge_result(self):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        n = len(self.edge_model.x_pairs)
        self.canvas.create_text(cx, cy - 80,
            text="Edge calibration complete  ✓",
            fill="#55ff88", font=("Helvetica", 28, "bold"))
        self.canvas.create_text(cx, cy - 20,
            text=f"Collected {n} edge correction samples.\n"
                  "Your personal gaze-to-edge mapping is ready.",
            fill="#aaaaaa", font=("Helvetica", 16), justify="center")
        accept_btn = tk.Button(self.root, text="✓  Start Gaze Mouse",
            bg="#1a6e3c", fg="white", font=("Helvetica", 18, "bold"),
            padx=20, pady=10, relief="flat", cursor="hand2",
            command=self._accept)
        self.canvas.create_window(cx - 140, cy + 70, window=accept_btn)
        retry_btn = tk.Button(self.root, text="↺  Redo Edge Trace",
            bg="#2a2a2a", fg="white", font=("Helvetica", 18),
            padx=20, pady=10, relief="flat", cursor="hand2",
            command=self._start_edge_intro)
        self.canvas.create_window(cx + 130, cy + 70, window=retry_btn)

    # ── Shared dot drawing ────────────────────────────────────────────────────

    def _draw_dot(self, px, py, radius, label, colour="#55aaff"):
        self.canvas.delete("all")
        self._clear_widgets()
        self.canvas.create_text(self.sw // 2, 36,
            text=label, fill="#666666", font=("Helvetica", 15))
        self.canvas.create_oval(
            px-radius, py-radius, px+radius, py+radius,
            outline=colour, width=2, fill="")
        self.canvas.create_oval(
            px-DOT_INNER, py-DOT_INNER, px+DOT_INNER, py+DOT_INNER,
            fill="white", outline="")
        self._animate_shrink(px, py, radius, DOT_INNER, time.time(), colour)

    def _animate_shrink(self, px, py, start_r, min_r, t0, colour="#55aaff"):
        elapsed = (time.time() - t0) * 1000
        frac    = min(elapsed / SHRINK_MS, 1.0)
        r       = start_r - frac * (start_r - min_r)
        self.canvas.delete("shrink_ring")
        self.canvas.create_oval(px-r, py-r, px+r, py+r,
            outline=colour, width=2, fill="", tags="shrink_ring")
        if frac < 1.0:
            self.root.after(16, self._animate_shrink,
                            px, py, start_r, min_r, t0, colour)

    # ── Accept / abort ────────────────────────────────────────────────────────

    def _accept(self):
        self.accepted = True
        self.root.destroy()

    def _abort(self):
        self.accepted = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.accepted, self.edge_model


# ─── Smooth Pursuit Calibration Window ───────────────────────────────────────

class CalibrationWindow:
    """
    Three-phase calibration (default):
      Phase 1: standard Tobii 5-point calibration
      Phase 2: smooth pursuit — follow a moving dot (figure-8, ~15 s)
               Collects hundreds of gaze→screen pairs across the full screen
      Phase 3: validation grid — 9-point accuracy check with error display
    Returns (accepted: bool, model: EdgeCorrectionModel)
    """

    PURSUIT_DURATION_S = 15.0   # how long the pursuit phase runs
    PURSUIT_FPS        = 60     # animation frame rate
    VALIDATION_POINTS  = [      # normalised (x, y) — 3×3 grid
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    ]
    VAL_DWELL_S = 1.5           # seconds per validation point

    def __init__(self, tracker):
        self.tracker    = tracker
        self.accepted   = False
        self.edge_model = EdgeCorrectionModel()

        self.root = tk.Tk()
        self.root.title("Tobii Calibration")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost",    True)

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._gaze_samples = []
        self._collecting   = False
        self._gaze_lock    = threading.Lock()
        self._latest_gaze  = None   # most recent valid gaze point

        self.root.bind("<Escape>", lambda e: self._abort())
        self._draw_intro()

    def _clear_widgets(self):
        for w in self.root.winfo_children():
            if isinstance(w, tk.Button):
                w.destroy()

    def _cx(self): return self.sw // 2
    def _cy(self): return self.sh // 2

    # ── Phase 0: intro ────────────────────────────────────────────────────────

    def _draw_intro(self):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        self.canvas.create_text(cx, cy - 140,
            text="Tobii Gaze Mouse — Calibration",
            fill="white", font=("Helvetica", 32, "bold"))
        self.canvas.create_text(cx, cy - 40,
            text="Three phases:\n\n"
                 "  Phase 1 — Standard 5-point Tobii calibration\n"
                 "  Phase 2 — Smooth pursuit: follow a moving dot for ~15 seconds\n"
                 "             (covers the full screen, gives hundreds of data points)\n"
                 "  Phase 3 — Validation: 9-point accuracy check\n"
                 "             (see errors before accepting)",
            fill="#aaaaaa", font=("Helvetica", 16), justify="center")
        self.canvas.create_text(cx, cy + 120,
            text="Click anywhere to begin Phase 1",
            fill="#55aaff", font=("Helvetica", 20))
        self.canvas.bind("<Button-1>", lambda e: self._start_tobii_calib())

    # ── Phase 1: Tobii calibration ────────────────────────────────────────────

    def _start_tobii_calib(self):
        self.canvas.unbind("<Button-1>")
        calib = tr.ScreenBasedCalibration(self.tracker)
        calib.enter_calibration_mode()
        threading.Thread(target=self._run_tobii_points,
                         args=(calib,), daemon=True).start()

    def _run_tobii_points(self, calib):
        total = len(TOBII_CALIB_POINTS)
        for i, (nx, ny) in enumerate(TOBII_CALIB_POINTS):
            px, py = int(nx * self.sw), int(ny * self.sh)
            self.root.after(0, self._draw_dot, px, py, DOT_RADIUS,
                            f"Phase 1 of 3 — Point {i+1} of {total}")
            time.sleep(SHRINK_MS / 1000)
            calib.collect_data(nx, ny)
            time.sleep(DWELL_MS / 1000)
        result = calib.compute_and_apply()
        calib.leave_calibration_mode()
        self.root.after(0, self._show_tobii_result, result)

    def _show_tobii_result(self, result):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        if result.status == tr.CALIBRATION_STATUS_SUCCESS:
            self.canvas.create_text(cx, cy - 60,
                text="Phase 1 complete  ✓",
                fill="#55ff88", font=("Helvetica", 28, "bold"))
            self.canvas.create_text(cx, cy,
                text="Now follow the moving dot with your eyes.",
                fill="#aaaaaa", font=("Helvetica", 16))
            btn = tk.Button(self.root, text="Continue to Smooth Pursuit →",
                bg="#1a4e8c", fg="white", font=("Helvetica", 18, "bold"),
                padx=20, pady=10, relief="flat", cursor="hand2",
                command=self._start_pursuit_intro)
            self.canvas.create_window(cx, cy + 70, window=btn)
        else:
            self.canvas.create_text(cx, cy - 60,
                text="Phase 1 failed — please retry",
                fill="#ff5555", font=("Helvetica", 28, "bold"))
            btn = tk.Button(self.root, text="↺  Retry Phase 1",
                bg="#2a2a2a", fg="white", font=("Helvetica", 18),
                padx=20, pady=10, relief="flat", cursor="hand2",
                command=self._draw_intro)
            self.canvas.create_window(cx, cy + 60, window=btn)

    # ── Phase 2: Smooth pursuit ───────────────────────────────────────────────

    def _start_pursuit_intro(self):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        self.canvas.create_text(cx, cy - 80,
            text="Phase 2 — Smooth Pursuit",
            fill="white", font=("Helvetica", 28, "bold"))
        self.canvas.create_text(cx, cy,
            text=f"A dot will move around the screen for {int(self.PURSUIT_DURATION_S)} seconds.\n"
                  "Follow it smoothly with your eyes — don't move your head.\n"
                  "Keep your eyes on the dot the whole time.",
            fill="#aaaaaa", font=("Helvetica", 16), justify="center")
        self.canvas.create_text(cx, cy + 100,
            text="Click anywhere to begin",
            fill="#55aaff", font=("Helvetica", 20))
        self.canvas.bind("<Button-1>", lambda e: self._start_pursuit())

    def _pursuit_pos(self, t):
        """Figure-8 (Lissajous) path, inset from screen edges."""
        margin_x = int(self.sw * 0.12)
        margin_y = int(self.sh * 0.12)
        rx = (self.sw - 2 * margin_x) / 2
        ry = (self.sh - 2 * margin_y) / 2
        # figure-8: x = sin(t), y = sin(2t)/2  (slow enough to track)
        speed = 2 * math.pi / self.PURSUIT_DURATION_S * 2
        x = self.sw / 2 + rx * math.sin(speed * t)
        y = self.sh / 2 + ry * math.sin(speed * t * 2) / 2
        return int(x), int(y)

    def _collect_gaze(self, gaze_data):
        l_pt = gaze_data["left_gaze_point_on_display_area"]
        r_pt = gaze_data["right_gaze_point_on_display_area"]
        lv   = gaze_data["left_gaze_point_validity"]
        rv   = gaze_data["right_gaze_point_validity"]
        pts  = []
        if lv and l_pt[0] == l_pt[0]: pts.append(l_pt)
        if rv and r_pt[0] == r_pt[0]: pts.append(r_pt)
        if pts:
            ax = sum(p[0] for p in pts) / len(pts)
            ay = sum(p[1] for p in pts) / len(pts)
            self._latest_gaze = (ax, ay)
            if self._collecting:
                with self._gaze_lock:
                    self._gaze_samples.append((ax, ay, time.time()))

    def _start_pursuit(self):
        self.canvas.unbind("<Button-1>")
        self._gaze_samples = []
        self._collecting   = True
        self.tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self._collect_gaze, as_dictionary=True)
        threading.Thread(target=self._run_pursuit, daemon=True).start()

    def _run_pursuit(self):
        t0    = time.time()
        frame = 0
        interval = 1.0 / self.PURSUIT_FPS
        while True:
            t = time.time() - t0
            if t >= self.PURSUIT_DURATION_S:
                break
            px, py = self._pursuit_pos(t)
            pct    = t / self.PURSUIT_DURATION_S * 100
            self.root.after(0, self._draw_pursuit_frame, px, py, pct)
            frame += 1
            next_t = t0 + frame * interval
            sleep  = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

        self._collecting = False
        self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._collect_gaze)

        # Build calibration model from pursuit data
        # Each gaze sample is paired with the dot position at that timestamp
        t0_abs = time.time() - self.PURSUIT_DURATION_S
        with self._gaze_lock:
            samples = list(self._gaze_samples)

        self.edge_model = EdgeCorrectionModel()
        t_start = samples[0][2] if samples else 0
        t_end   = samples[-1][2] if samples else 0
        dur     = t_end - t_start if t_end > t_start else 1
        lag_s   = PURSUIT_LAG_MS / 1000.0

        for (gx, gy, ts) in samples:
            # Pair gaze with where the dot WAS lag_s seconds earlier
            # (eyes trail a moving target by ~100-150ms)
            t_norm = ((ts - lag_s) - t_start) / dur * self.PURSUIT_DURATION_S
            t_norm = max(0.0, t_norm)
            dot_px, dot_py = self._pursuit_pos(t_norm)
            true_nx = dot_px / self.sw
            true_ny = dot_py / self.sh
            self.edge_model.add_sample(true_nx, true_ny, gx, gy)

        self.edge_model.fit()
        n = len(samples)
        print(f"  Pursuit collected {n} gaze samples.")
        self.root.after(0, self._show_pursuit_result, n)

    def _draw_pursuit_frame(self, px, py, pct):
        self.canvas.delete("all")
        r = DOT_RADIUS
        self.canvas.create_text(self.sw // 2, 30,
            text=f"Phase 2 of 3 — Follow the dot  ({pct:.0f}%)",
            fill="#666666", font=("Helvetica", 14))
        self.canvas.create_oval(px-r, py-r, px+r, py+r,
            outline="#55aaff", width=2, fill="")
        self.canvas.create_oval(px-DOT_INNER, py-DOT_INNER,
                                px+DOT_INNER, py+DOT_INNER,
            fill="white", outline="")

    def _show_pursuit_result(self, n):
        self.canvas.delete("all")
        self._clear_widgets()
        cx, cy = self._cx(), self._cy()
        self.canvas.create_text(cx, cy - 60,
            text="Phase 2 complete  ✓",
            fill="#55ff88", font=("Helvetica", 28, "bold"))
        self.canvas.create_text(cx, cy,
            text=f"Collected {n} gaze samples.\nNow verifying accuracy across 9 screen positions.",
            fill="#aaaaaa", font=("Helvetica", 16), justify="center")
        btn = tk.Button(self.root, text="Continue to Validation →",
            bg="#1a4e8c", fg="white", font=("Helvetica", 18, "bold"),
            padx=20, pady=10, relief="flat", cursor="hand2",
            command=self._start_validation)
        self.canvas.create_window(cx, cy + 80, window=btn)

    # ── Phase 3: Validation grid ──────────────────────────────────────────────

    def _start_validation(self):
        self.canvas.delete("all")
        self._clear_widgets()
        self._val_errors  = []   # [(true_x, true_y, err_px), ...]
        self._val_vectors = []   # [(true_px, true_py, gaze_px, gaze_py), ...]
        self.tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self._collect_gaze, as_dictionary=True)
        threading.Thread(target=self._run_validation, daemon=True).start()

    def _run_validation(self):
        total = len(self.VALIDATION_POINTS)
        for i, (nx, ny) in enumerate(self.VALIDATION_POINTS):
            margin = 40
            px = int(nx * (self.sw - 2*margin)) + margin
            py = int(ny * (self.sh - 2*margin)) + margin
            self.root.after(0, self._draw_dot, px, py, DOT_RADIUS,
                            f"Phase 3 of 3 — Validation point {i+1} of {total}",
                            "#ff9933")
            time.sleep(0.4)   # let eyes land

            # Collect samples
            with self._gaze_lock:
                self._gaze_samples = []
            self._collecting = True
            time.sleep(self.VAL_DWELL_S)
            self._collecting = False

            with self._gaze_lock:
                samps = list(self._gaze_samples)

            if samps:
                avg_gx = sum(s[0] for s in samps) / len(samps)
                avg_gy = sum(s[1] for s in samps) / len(samps)
                # Apply correction
                corr_nx, corr_ny = self.edge_model.correct(avg_gx, avg_gy)
                corr_px = int(corr_nx * self.sw)
                corr_py = int(corr_ny * self.sh)
                err = math.hypot(corr_px - px, corr_py - py)
                self._val_errors.append((nx, ny, err))
                self._val_vectors.append((px, py, corr_px, corr_py))

        self._collecting = False
        self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._collect_gaze)
        self.root.after(0, self._show_validation_result)

    def _show_validation_result(self):
        self.canvas.delete("all")
        self._clear_widgets()

        # Draw error vectors on a dark grid
        for (true_px, true_py, gaze_px, gaze_py) in self._val_vectors:
            # True target
            self.canvas.create_oval(
                true_px-8, true_py-8, true_px+8, true_py+8,
                outline="#55ff88", width=2, fill="")
            # Corrected gaze landing
            self.canvas.create_oval(
                gaze_px-5, gaze_py-5, gaze_px+5, gaze_py+5,
                fill="#ff5555", outline="")
            # Error vector
            self.canvas.create_line(
                true_px, true_py, gaze_px, gaze_py,
                fill="#ffaa00", width=1, dash=(4, 3))

        cx, cy = self._cx(), self._cy()
        if self._val_errors:
            avg_err = sum(e[2] for e in self._val_errors) / len(self._val_errors)
            max_err = max(e[2] for e in self._val_errors)
            self.canvas.create_text(cx, 40,
                text=f"Validation complete  —  avg error: {avg_err:.0f} px   "
                     f"max: {max_err:.0f} px",
                fill="white", font=("Helvetica", 16, "bold"))
            self.canvas.create_text(cx, 70,
                text="Green circle = target   Red dot = where gaze landed   "
                     "Orange line = error",
                fill="#888888", font=("Helvetica", 13))
        else:
            self.canvas.create_text(cx, 50,
                text="No validation data collected.",
                fill="#ff5555", font=("Helvetica", 16))

        accept_btn = tk.Button(self.root, text="✓  Accept & Start Gaze Mouse",
            bg="#1a6e3c", fg="white", font=("Helvetica", 18, "bold"),
            padx=20, pady=10, relief="flat", cursor="hand2",
            command=self._accept)
        self.canvas.create_window(cx - 160, self.sh - 60, window=accept_btn)
        retry_btn = tk.Button(self.root, text="↺  Redo Pursuit",
            bg="#2a2a2a", fg="white", font=("Helvetica", 18),
            padx=20, pady=10, relief="flat", cursor="hand2",
            command=self._start_pursuit_intro)
        self.canvas.create_window(cx + 130, self.sh - 60, window=retry_btn)

    # ── Shared dot drawing ────────────────────────────────────────────────────

    def _draw_dot(self, px, py, radius, label, colour="#55aaff"):
        self.canvas.delete("all")
        self._clear_widgets()
        self.canvas.create_text(self.sw // 2, 36,
            text=label, fill="#666666", font=("Helvetica", 15))
        self.canvas.create_oval(
            px-radius, py-radius, px+radius, py+radius,
            outline=colour, width=2, fill="")
        self.canvas.create_oval(
            px-DOT_INNER, py-DOT_INNER, px+DOT_INNER, py+DOT_INNER,
            fill="white", outline="")
        self._animate_shrink(px, py, radius, DOT_INNER, time.time(), colour)

    def _animate_shrink(self, px, py, start_r, min_r, t0, colour="#55aaff"):
        elapsed = (time.time() - t0) * 1000
        frac    = min(elapsed / SHRINK_MS, 1.0)
        r       = start_r - frac * (start_r - min_r)
        self.canvas.delete("shrink_ring")
        self.canvas.create_oval(px-r, py-r, px+r, py+r,
            outline=colour, width=2, fill="", tags="shrink_ring")
        if frac < 1.0:
            self.root.after(16, self._animate_shrink,
                            px, py, start_r, min_r, t0, colour)

    # ── Accept / abort ────────────────────────────────────────────────────────

    def _accept(self):
        self.accepted = True
        self.root.destroy()

    def _abort(self):
        self.accepted = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.accepted, self.edge_model


# ─── Gaze Mouse + Blink Controller ───────────────────────────────────────────

class GazeMouseController:

    def __init__(self, tracker, edge_model, ui_scanner):
        self.tracker     = tracker
        self.edge_model  = edge_model
        self.ui_scanner  = ui_scanner
        self.sw, self.sh = pyautogui.size()

        self.gaze_buffer = collections.deque(maxlen=GAZE_SMOOTH_SAMPLES)
        self.last_gaze   = (self.sw // 2, self.sh // 2)

        mx, my = pyautogui.position()
        self.smooth_x = float(mx)
        self.smooth_y = float(my)

        # Dwell
        self.dwell_history = collections.deque()
        self.dwell_locked  = False
        self.dwell_pos     = None

        # Precision mode
        self.precision_history = collections.deque()
        self.precision_active  = False

        # Blink
        self.in_blink       = False
        self.blink_start_ts = None
        self.last_blink_end = None
        self.last_click_ts  = 0
        self.click_lock_pos = None
        self.lock           = threading.Lock()

        self._running        = True
        self._last_display_x = self.sw // 2
        self._last_display_y = self.sh // 2

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dist(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    # ── edge gravity (screen edges) ───────────────────────────────────────────

    def _apply_edge_gravity(self, x, y):
        gx_px = int(self.sw * EDGE_GRAVITY_ZONE)
        gy_px = int(self.sh * EDGE_GRAVITY_ZONE)
        nx, ny = float(x), float(y)

        # Cap t at 0.75 so pull never reaches 100% — cursor can always escape
        MAX_T = 0.50

        if x < gx_px:
            t = min((gx_px - x) / gx_px, MAX_T)
            nx = x - x * t * EDGE_GRAVITY_STRENGTH
        elif x > self.sw - gx_px:
            t = min((x - (self.sw - gx_px)) / gx_px, MAX_T)
            nx = x + (self.sw - 1 - x) * t * EDGE_GRAVITY_STRENGTH

        if y < gy_px:
            t = min((gy_px - y) / gy_px, MAX_T)
            ny = y - y * t * EDGE_GRAVITY_STRENGTH
        elif y > self.sh - gy_px:
            t = min((y - (self.sh - gy_px)) / gy_px, MAX_T)
            ny = y + (self.sh - 1 - y) * t * EDGE_GRAVITY_STRENGTH

        return int(nx), int(ny)

    # ── UI element gravity (stronger) ─────────────────────────────────────────

    def _apply_ui_gravity(self, x, y):
        """Pull cursor toward the nearest UI element if within radius."""
        if self.ui_scanner is None:
            return x, y
        targets = self.ui_scanner.get_targets()
        if not targets:
            return x, y

        best_d = UI_GRAVITY_RADIUS_PX + 1
        best_t = None
        for (tx, ty, _label) in targets:
            d = self._dist(x, y, tx, ty)
            if d < best_d:
                best_d = d
                best_t = (tx, ty)

        if best_t is None:
            return x, y

        # Lerp toward the element centre — stronger the closer we are
        t      = 1.0 - (best_d / UI_GRAVITY_RADIUS_PX)   # 0 at edge, 1 at centre
        pull   = t * UI_GRAVITY_STRENGTH
        nx     = int(x + (best_t[0] - x) * pull)
        ny     = int(y + (best_t[1] - y) * pull)
        return nx, ny

    # ── dwell detection ───────────────────────────────────────────────────────

    def _update_dwell(self, gx, gy, now_ms):
        self.dwell_history.append((now_ms, gx, gy))
        cutoff = now_ms - DWELL_TIME_MS
        while self.dwell_history and self.dwell_history[0][0] < cutoff:
            self.dwell_history.popleft()
        if len(self.dwell_history) < 3:
            return
        xs = [p[1] for p in self.dwell_history]
        ys = [p[2] for p in self.dwell_history]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        all_close = all(
            self._dist(p[1], p[2], cx, cy) <= DWELL_RADIUS_PX
            for p in self.dwell_history)
        if all_close:
            self.dwell_locked = True
            self.dwell_pos    = (int(cx), int(cy))
            self.smooth_x, self.smooth_y = cx, cy
        else:
            self.dwell_locked = False
            self.dwell_pos    = None

    # ── gaze callback ─────────────────────────────────────────────────────────

    def on_gaze(self, gaze_data):
        now_ms  = time.time() * 1000
        l_pt    = gaze_data["left_gaze_point_on_display_area"]
        r_pt    = gaze_data["right_gaze_point_on_display_area"]
        l_valid = gaze_data["left_gaze_point_validity"]
        r_valid = gaze_data["right_gaze_point_validity"]

        pts = []
        if l_valid and l_pt[0] == l_pt[0]: pts.append(l_pt)
        if r_valid and r_pt[0] == r_pt[0]: pts.append(r_pt)

        if pts:
            ax = sum(p[0] for p in pts) / len(pts)
            ay = sum(p[1] for p in pts) / len(pts)
            ax = max(0.0, min(ax, 1.0))
            ay = max(0.0, min(ay, 1.0))

            # 1. Personal edge correction (from edge trace calibration)
            ax, ay = self.edge_model.correct(ax, ay)
            ay = ay + GAZE_Y_OFFSET          # vertical bias correction
            ax = max(0.0, min(ax, 1.0))
            ay = max(0.0, min(ay, 1.0))

            # Raw pixel coords — no gravity yet, buffer raw for smoothing
            gx = int(ax * (self.sw - 1))
            gy = int(ay * (self.sh - 1))
            self.gaze_buffer.append((gx, gy))

            if self.gaze_buffer:
                # Rolling average over raw gaze — kills sample-to-sample jitter
                rx = int(sum(g[0] for g in self.gaze_buffer) / len(self.gaze_buffer))
                ry = int(sum(g[1] for g in self.gaze_buffer) / len(self.gaze_buffer))
                self.last_gaze = (rx, ry)

                # Precision mode: check if gaze has been stable recently
                self.precision_history.append((now_ms, rx, ry))
                cutoff = now_ms - PRECISION_STABLE_MS
                while self.precision_history and self.precision_history[0][0] < cutoff:
                    self.precision_history.popleft()
                if len(self.precision_history) >= 3:
                    xs = [p[1] for p in self.precision_history]
                    ys = [p[2] for p in self.precision_history]
                    cx = sum(xs) / len(xs)
                    cy = sum(ys) / len(ys)
                    stable = all(
                        self._dist(p[1], p[2], cx, cy) <= PRECISION_STABLE_PX
                        for p in self.precision_history)
                    self.precision_active = stable
                else:
                    self.precision_active = False

                # Exponential smoothing — tighter in precision mode
                smoothing = PRECISION_SMOOTHING if self.precision_active else MOUSE_SMOOTHING
                self.smooth_x += (rx - self.smooth_x) * (1.0 - smoothing)
                self.smooth_y += (ry - self.smooth_y) * (1.0 - smoothing)
                sx = int(self.smooth_x)
                sy = int(self.smooth_y)

                # Store final position for blink snapshot
                self._last_display_x = sx
                self._last_display_y = sy

                # Dwell detection runs on the smoothed+gravity position
                if self.click_lock_pos is None:
                    self._update_dwell(sx, sy, now_ms)

                # Move mouse
                if self.click_lock_pos is not None:
                    pass  # frozen during blink sequence
                elif self.dwell_locked and self.dwell_pos is not None:
                    pyautogui.moveTo(*self.dwell_pos)
                else:
                    mcx, mcy = pyautogui.position()
                    if abs(sx-mcx) > MOVE_THRESHOLD_PX or abs(sy-mcy) > MOVE_THRESHOLD_PX:
                        pyautogui.moveTo(sx, sy)

        # ── Blink state machine ────────────────────────────────────────────────
        both_invalid = not l_valid and not r_valid

        with self.lock:
            if both_invalid and not self.in_blink:
                self.in_blink       = True
                self.blink_start_ts = now_ms
                if self.click_lock_pos is None:
                    if self.dwell_locked and self.dwell_pos:
                        self.click_lock_pos = self.dwell_pos
                    else:
                        # Use post-gravity smoothed position for accurate snapshot
                        self.click_lock_pos = (self._last_display_x, self._last_display_y)

            elif not both_invalid and self.in_blink:
                self.in_blink = False
                duration      = now_ms - self.blink_start_ts

                if BLINK_MIN_MS <= duration <= BLINK_MAX_MS:
                    if (self.last_blink_end is not None and
                            (self.blink_start_ts - self.last_blink_end) <= DOUBLE_BLINK_WINDOW_MS):
                        self._click()
                        self.last_blink_end = None
                        self.click_lock_pos = None
                    else:
                        self.last_blink_end = now_ms
                        threading.Timer(
                            DOUBLE_BLINK_WINDOW_MS / 1000,
                            self._release_lock_if_expired,
                            args=(now_ms,)
                        ).start()
                else:
                    self.last_blink_end = None
                    self.click_lock_pos = None

    def _click(self):
        now_ms = time.time() * 1000
        if now_ms - self.last_click_ts < COOLDOWN_MS:
            return
        self.last_click_ts = now_ms
        x, y = self.click_lock_pos if self.click_lock_pos else \
               (int(self.smooth_x), int(self.smooth_y))
        x = max(0, min(x, self.sw - 1))
        y = max(0, min(y, self.sh - 1))
        tag = " [dwell]" if self.dwell_locked else ""
        print(f"  ◉ click → ({x}, {y}){tag}")
        pyautogui.click(x, y)

    def _release_lock_if_expired(self, lock_set_ts):
        with self.lock:
            if self.last_blink_end == lock_set_ts:
                self.click_lock_pos = None
                self.last_blink_end = None

    def start(self):
        self.tracker.subscribe_to(
            tr.EYETRACKER_GAZE_DATA, self.on_gaze, as_dictionary=True)
        print("\n──────────────────────────────────────────────────────")
        print("  Gaze Mouse LIVE")
        print("  Eyes move       →  mouse follows (edge-corrected)")
        print("  Gaze + hold     →  dwell-lock to target")
        print("  Near UI element →  stronger gravity snap")
        print("  Double-blink    →  left-click")
        print("  Ctrl+C / top-left corner  →  quit")
        print("──────────────────────────────────────────────────────\n")
        if HAS_PYWINAUTO:
            print(f"  UI scanner active — rescanning every {UI_SCAN_INTERVAL_S}s")
        else:
            print("  UI scanner disabled (install pywinauto to enable)")

    def stop(self):
        self._running = False
        self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.on_gaze)
        if self.ui_scanner:
            self.ui_scanner.stop()
        print("Stopped.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tobii Gaze Mouse")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy edge-trace calibration instead of smooth pursuit")
    args = parser.parse_args()

    print("Searching for Tobii eye trackers …")
    trackers = tr.find_all_eyetrackers()
    if not trackers:
        print("No Tobii tracker found. Is the device connected and Tobii software running?")
        sys.exit(1)

    tracker = trackers[0]
    print(f"Found: {tracker.model}  (S/N {tracker.serial_number})\n")

    if args.legacy:
        print("Using legacy edge-trace calibration.")
        accepted, edge_model = LegacyCalibrationWindow(tracker).run()
    else:
        accepted, edge_model = CalibrationWindow(tracker).run()

    if not accepted:
        print("Calibration cancelled.")
        sys.exit(0)

    print(f"Calibration complete — {len(edge_model._samples)} samples collected.")

    sw, sh = pyautogui.size()
    ui_scanner = UIScanner(sw, sh) if HAS_PYWINAUTO else None

    controller = GazeMouseController(tracker, edge_model, ui_scanner)
    controller.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
