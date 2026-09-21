"""Interactive NeuroPAL GUI, with two tabs over one loaded volume.

**Processing** balances the colour channels: no per-channel normalization, just divide
each channel by an overall factor, leave the relative brightness otherwise intact, and
recombine. It writes the corrected volumes.

**Labeling** shows the same volume for locating neurons, one z slice at a time rather than
a projection, and deliberately exposes nothing that can write a volume.

Orientation flips drive the panels and are baked into what Processing writes, with
neuropal_info.txt recording which ones were applied. The Labeling tab always records a
neuron's position in the coordinates of the file that is loaded, so labelling the corrected
volume with the flips off gives coordinates in that volume's own frame.

Each channel's brightness is expressed as a percentage: the reciprocal of the factor the
channel is divided by.

    factor = 100 / percentage        percentage = 100 / factor

so 100% leaves a channel untouched (factor 1.0), 33.3% is factor 3.0 (dims it), and 200%
is factor 0.5 (brightens it). Showing all three channels together at 100% is the raw
volume.

Run this through run_neuropal_gui_example.py.
"""

import csv
import itertools
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import matplotlib.patheffects as patheffects
import numpy as np
import nrrd
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Channel order of the 4th axis of NeuroPAL.nrrd, with the marker each one holds.
CHANNELS = [("R", "Red channel (mNeptune)", 0),
            ("G", "Green channel (OFP)", 1),
            ("B", "Blue channel (BFP)", 2)]

# What the panels show. The long names above are what neuropal_info.txt records, so they
# stay put; these just have to fit a narrow column next to a slider.
CHANNEL_SHORT = {"R": "R (mNeptune)", "G": "G (OFP)", "B": "B (BFP)"}

# Companion volumes, shown as overlays and copied into the output folder.
COMPANION_VOLUMES = ["all_red", "neuron_rois"]

# Everything either tab produces goes in this one folder, beside the NeuroPAL file unless
# the launcher names another: the rescaled volume, copies of the companions, the
# scale-factor record, and the label table.
CORRECTED_DIR_NAME = "corrected_nrrds_and_labels"

# The volume axes the display orientation can flip, as (volume axis, key, what it turns
# round on screen). x is in plane in both panels, so flipping it turns both round; y is
# only in plane in the lateral view and z only in the coronal.
FLIPS = [(0, "x", "lateral left-right"),
         (1, "y", "lateral up-down"),
         (2, "z", "coronal up-down")]

# Anatomical letter for each end of each volume axis, in axis order. An orientation code
# names the direction each axis points *towards* as its index increases, so "LPS" means
# +x -> Left, +y -> Posterior, +z -> Superior. This is the usual RAI/LPS convention.
ORIENTATION_AXES = [("L", "R"), ("P", "A"), ("S", "I")]
OPPOSITE_LETTER = {a: b for pair in ORIENTATION_AXES for a, b in (pair, pair[::-1])}

# An odd number of changed letters reflects the specimen rather than rotating it, and no
# rotation undoes a reflection -- that is what swaps an animal's left and right. Those codes
# are not offered at all: the viewer turns a specimen round, it does not restructure it.


def source_orientation(header):
    """The orientation code the file itself declares, e.g. "LPS".

    Read from the NRRD `space` field, with a negative diagonal in `space directions`
    flipping that axis's letter. Falls back to LPS when the field is missing or unreadable.
    """
    space = str(header.get("space", "")).lower().replace("-", " ")
    words = space.split()
    if len(words) == 3:
        code = "".join(word[0].upper() for word in words)
    else:
        code = "".join(c for c in space.upper() if c in OPPOSITE_LETTER)[:3]
    if len(code) != 3 or any(c not in OPPOSITE_LETTER for c in code):
        code = "LPS"

    directions = header.get("space directions")
    if directions is not None:
        directions = np.asarray(directions, dtype=float)[:3, :3]
        if np.isfinite(directions).all():
            code = "".join(OPPOSITE_LETTER[c] if directions[i][i] < 0 else c
                           for i, c in enumerate(code))
    return code


def orientation_options(source):
    """Every code reachable from `source` by flipping axes, as (code, flips, mirrored).

    Only sign changes, not axis swaps: the panels are built around x being the long axis,
    so a code that permutes the axes is not something this viewer can show.
    """
    options = []
    for letters in itertools.product(*((c, OPPOSITE_LETTER[c]) for c in source)):
        flips = tuple(letters[i] != source[i] for i in range(3))
        options.append(("".join(letters), flips, sum(flips) % 2 == 1))
    return options


def flips_for_orientation(source, code):
    """Which axes to flip to show `source` data as `code`."""
    return {key: code[axis] != source[axis] for axis, key, _ in FLIPS}


def orientation_after_flips(source, flips):
    """The code `source` data ends up in once `flips` are applied."""
    return "".join(OPPOSITE_LETTER[letter] if flips.get(key) else letter
                   for (_, key, _), letter in zip(FLIPS, source))


SLIDER_MAX = 200.0          # percent; the entry boxes clamp to this too
DISPLAY_PERCENTILE = 99.9   # of the raw MIP, used as the display white point

# Which end of each panel's vertical axis row 0 is drawn at, keyed by the view's volume
# axis. This follows ITK-SNAP, which does not pick an image convention per panel at all: it
# draws in anatomical space and puts the anatomically-up direction at the top of every
# view. Under LPS that works out differently per panel, because +y points posterior while
# +z points superior --
#
#   lateral  (slices z, shows x/y): screen-up = Anterior = -y  ->  y down  -> "upper"
#   coronal  (slices y, shows x/z): screen-up = Superior = +z  ->  z up    -> "lower"
#
# It matters most on the coronal panel, whose vertical axis is the left-right axis: with
# "upper" there, a neuron on one side appears at the top here and the bottom in ITK-SNAP,
# which is exactly where an L/R call is easiest to get wrong.
#
# Display-only: it flips the image on screen and has no effect on the saved volumes.
DISPLAY_ORIGIN = "upper"
VIEW_ORIGIN = {2: "upper", 1: "lower"}

# Crosshair linking the views, ITK-SNAP style: one 3D cursor, each panel showing the two
# coordinates it has in plane. The x coordinate is shared by both panels, so clicking in
# either moves the other's vertical line with it.
CURSOR_COLOR = "#5b7cfa"

# The two views, as (volume axis, name). The axis is the one projected out for a MIP, or
# stepped through for a slice: the lateral view shows (X, Y) at some z, the coronal shows
# (X, Z) at some y. Both keep x in plane, so they share their x axis and stack naturally --
# the same pair ITK-SNAP shows.
VIEWS = [(2, "Lateral"), (1, "Coronal")]
AXIS_LETTER = {0: "x", 1: "y", 2: "z"}

# Overlays offered by both tabs, as (key, companion volume, label, hotkey, default
# opacity %, shown by default). A tab can start one switched off through `overlays_off`.
# Display only, like everything else about them.
OVERLAYS = [("red", "all_red", "all_red (grey)", "a", 50.0, False),
            ("seg", "neuron_rois", "neuron_rois", "s", 50.0, True)]

# A wide, high-contrast colormap, so neighbouring ROI ids are easy to tell apart. Scaled
# against the whole label range, not per slice, so an ROI does not change colour as you
# scroll.
SEGMENTATION_CMAP = "nipy_spectral"

# The label table the Labeling tab writes, named <dataset folder>_labels.csv and kept with
# the corrected volumes. Column order is the file's contract -- read it back with anything,
# but do not reorder these.
LABEL_COLUMNS = ["Neuron Class", "Coordinates", "ROI ID", "Confidence",
                 "Second Candidate", "Second Candidate Confidence", "Notes"]
NOTES_MAX_CHARS = 300
CONFIDENCE_RANGE = (1, 5)
# A flag rather than a score: the ROI is two neurons merged by the segmentation, so it is
# labelled but should not be used for analysis. Sorts below every real score, so the
# threshold hides these until it is dragged all the way down.
CONFIDENCE_SPLIT_ROI = -1
CONFIDENCE_SPLIT_NOTE = "or -1 for a split ROI (merged at segmentation)"
# The range the "show these names" threshold can be dragged over: every value that can be
# recorded, so nothing that can be saved is unreachable by the filter.
CONFIDENCE_THRESHOLD_RANGE = (CONFIDENCE_SPLIT_ROI, CONFIDENCE_RANGE[1])
CONFIDENCE_THRESHOLD_DEFAULT = 4        # what both tabs open on
LABEL_HOTKEY = "n"                      # toggles the names, like "a" and "s" do overlays

# Saved neuron names drawn over the views. White over a thin black outline, so the text
# stays legible against both bright neurons and black background.
LABEL_TEXT_COLOR = "white"
LABEL_FONT_SIZE = 5


def default_output_dir(src_path):
    """Where both tabs write, unless the launcher names somewhere else."""
    return os.path.join(os.path.dirname(os.path.abspath(src_path)), CORRECTED_DIR_NAME)


def labels_path_for(src_path, out_dir=None):
    """<dataset folder>_labels.csv, alongside the corrected volumes it describes.

    Named after the dataset folder (e.g. 2023-01-06-08_labels.csv) and written into the output
    folder, so every output of both tabs lands in one place.

    Re-loading the corrected volumes to label them puts the source *inside* the output
    folder; the dataset is then the folder above it, not CORRECTED_DIR_NAME itself, so both
    passes over a dataset write to the same table.
    """
    folder = os.path.dirname(os.path.abspath(src_path))
    if os.path.basename(folder) == CORRECTED_DIR_NAME:
        folder = os.path.dirname(folder)
    return os.path.join(out_dir or default_output_dir(src_path),
                        f"{os.path.basename(folder)}_labels.csv")


def read_labels(path):
    """Load the label table into {roi id: row}. A missing file is simply an empty table."""
    rows = {}
    if not os.path.exists(path):
        return rows
    # newline="" is the csv module's contract; utf-8 so a table written on one platform
    # reads on another -- Windows would otherwise default to cp1252 and choke on a note
    # containing anything outside it.
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows[int(row["ROI ID"])] = {c: (row.get(c) or "") for c in LABEL_COLUMNS}
            except (TypeError, ValueError):
                continue        # a row without a usable ROI ID is not addressable
    return rows


def write_labels(path, rows):
    """Rewrite the whole table: named rows by ROI id, then the unnamed ones.

    Rewriting rather than appending is what keeps one row per ROI: re-saving an ROI edits
    its row in place instead of leaving two rows that disagree. Rows whose Neuron Class is
    blank are held for an ROI that has not been identified yet, so they go at the end and
    leave the identified list clean to read.
    """
    def order(roi_id):
        named = bool((rows[roi_id].get("Neuron Class") or "").strip())
        return (0 if named else 1, roi_id)

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)      # labelling may run before an image is saved
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for roi_id in sorted(rows, key=order):
            writer.writerow(rows[roi_id])


# --------------------------------------------------------------------------------------
# Volume handling
# --------------------------------------------------------------------------------------

def apply_flips(arr, fx, fy, fz=False):
    """Flip an (X, Y, Z, ...) array along any of its first three axes.

    Works for both the (X, Y, Z, C) NeuroPAL volume and the (X, Y, Z) companions. Returns
    a contiguous copy and preserves dtype (uint16 labels stay uint16). Used for the
    on-screen orientation only -- what gets written keeps the source's coordinates.
    """
    if fx:
        arr = arr[::-1]
    if fy:
        arr = arr[:, ::-1]
    if fz:
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def load_neuropal(path):
    """Read a NeuroPAL .nrrd and return (data, header). Kept unflipped and unscaled.

    Flips are re-derived from this pristine copy every time, so toggling them in the GUI
    never stacks.
    """
    data, header = nrrd.read(path)
    if data.ndim != 4 or data.shape[-1] < 3:
        raise ValueError(f"expected an (X, Y, Z, C>=3) volume, got shape {data.shape}")
    return data, header


def pct_to_factor(pct):
    """Brightness percentage -> the division factor. 0% -> inf (channel off)."""
    return float("inf") if pct <= 0 else 100.0 / pct


def factor_to_pct(factor):
    """Division factor -> brightness percentage."""
    return 0.0 if factor == float("inf") else 100.0 / factor


def rescale_channels(img, percentages):
    """Divide each channel of an (X, Y, Z, C) volume by its factor; nothing else touched.

    `percentages` maps channel key -> brightness percentage; a channel left at 100% passes
    through untouched.
    """
    out = img.astype(np.float32).copy()
    for key, _, idx in CHANNELS:
        factor = pct_to_factor(percentages[key])
        out[..., idx] = 0.0 if factor == float("inf") else out[..., idx] / factor
    return out


DEFAULT_POINTS = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]


def apply_transfer(x, points=None):
    """ITK-SNAP's curve-based contrast adjustment, on intensities normalized to [0, 1].

    `points` are the control points as (intensity, index into color map), both in [0, 1]
    and ordered by intensity. Between them the curve is a monotone cubic (PCHIP), which
    is what gives ITK-SNAP's smooth shoulder without ever letting the curve overshoot or
    turn back on itself; outside them it is flat at the end points' values.
    """
    points = points or DEFAULT_POINTS
    xs = np.array([px for px, _ in points], dtype=float)
    ys = np.array([py for _, py in points], dtype=float)

    out = PchipInterpolator(xs, ys)(np.clip(x, xs[0], xs[-1]))
    out = np.where(x <= xs[0], ys[0], out)
    out = np.where(x >= xs[-1], ys[-1], out)
    return np.clip(out, 0.0, 1.0)


def compose_plane(plane, percentages, visible, ref, points=None):
    """Turn one (X, V, 3) plane into a displayable RGB image.

    `ref` is computed once from the raw volume, so moving a slider changes only that
    channel's brightness on screen instead of renormalizing the whole composite.

    One curve is shared by the three channels, so their relative brightness on screen
    still reflects the data -- a per-channel curve would silently rebalance the colors
    that the whole point of this GUI is to judge.
    """
    plane = plane.astype(np.float32)
    disp = np.zeros_like(plane)
    for key, _, idx in CHANNELS:
        if not visible[key]:
            continue
        factor = pct_to_factor(percentages[key])
        if factor == float("inf"):
            continue
        disp[..., idx] = apply_transfer(plane[..., idx] / factor / ref, points)
    return disp


def composite_mip(img, percentages, visible, ref, points=None, axis=2):
    """Maximum intensity projection along `axis`, through the contrast curve."""
    return compose_plane(img[..., :3].max(axis=axis), percentages, visible, ref, points)


def composite_slice(img, percentages, visible, ref, points=None, axis=2, index=0):
    """The single slice at `index` along `axis`, through the same contrast curve.

    Same treatment as composite_mip so the two tabs are directly comparable: the slice is
    only ever a different set of voxels, never a different scaling.
    """
    index = int(np.clip(index, 0, img.shape[axis] - 1))
    return compose_plane(np.take(img[..., :3], index, axis=axis),
                         percentages, visible, ref, points)


def display_reference(img):
    """White point for the on-screen composite: a high percentile of the raw MIP.

    A percentile rather than the max, so one hot pixel cannot drive the whole image black.
    Shared by both tabs, so a slice and a projection of the same volume stay comparable.
    """
    mip = img[..., :3].max(axis=2).astype(np.float32)
    ref = float(np.percentile(mip, DISPLAY_PERCENTILE))
    return ref if ref > 0 else float(mip.max())


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

def resolve_companion_paths(src_path, all_red_path=None, neuron_rois_path=None):
    """Locate the companion volumes: explicit paths if given, else beside the NeuroPAL file.

    Passing a path only for some of them is fine -- the rest still fall back to the
    NeuroPAL file's own directory, which is the usual dataset layout.
    """
    explicit = {"all_red": all_red_path, "neuron_rois": neuron_rois_path}
    src_dir = os.path.dirname(os.path.abspath(src_path))
    return {name: (explicit[name] if explicit.get(name)
                   else os.path.join(src_dir, f"{name}.nrrd"))
            for name in COMPANION_VOLUMES}


def write_info_file(path, src_path, percentages, flips, source_code="LPS"):
    """Write neuropal_info.txt recording the orientation saved and the channel factors.

    The orientation is baked into the volumes beside this file, so the record has to name
    it: without it there is nothing to say which way round the saved volume is. It is given
    as an anatomical code, the same one the GUI's orientation box shows.
    """
    saved_code = orientation_after_flips(source_code, flips)
    mirrored = sum(1 for _, key, _ in FLIPS if flips.get(key)) % 2 == 1
    applied = [label for _, key, label in FLIPS if flips.get(key)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"Source: {os.path.abspath(src_path)}\n")
        fh.write(f"Source orientation: {source_code}\n")
        fh.write(f"Saved orientation:  {saved_code}"
                 + ("   (mirrored relative to the source: left and right are swapped)\n"
                    if mirrored else "   (a rotation of the source)\n"))
        fh.write("Axes turned round to get there: "
                 + (", ".join(applied) if applied else "none") + "\n")
        fh.write("Scale factors (channel divided by the division factor):\n")
        for key, label, _ in CHANNELS:
            pct = percentages[key]
            factor = pct_to_factor(pct)
            factor_str = "inf (channel off)" if factor == float("inf") else f"{factor:.6g}"
            fh.write(f"{label}: {pct:.6g}%, division factor {factor_str}\n")


def save_corrected(src_path, raw_data, percentages, out_dir, flips=None,
                   source_code="LPS", companion_paths=None):
    """Write the output folder: the rescaled volume, the companions, and the record.

    Contents:
      NeuroPAL_corrected.nrrd     flipped and rescaled
      all_red_corrected.nrrd      flipped only -- a single channel has nothing to rebalance
      neuron_rois_corrected.nrrd  flipped only -- integer labels, never scaled
      neuropal_info.txt           the flips applied, and the percentages used

    The display flips are applied here, so the saved folder is in the orientation that was
    on screen and everything in it stays voxel-aligned with everything else. `neuropal_info`
    records which flips those were, so the orientation is never a guess. Load the corrected
    volume back with the flips off and its own coordinates are what the Labeling tab
    records.

    `raw_data` is the pristine volume and the flips are re-applied from it here, so they
    can never stack. Only the channel factors are applied to the intensities -- no
    normalization, so the relative brightness between channels is exactly what was dialed
    in. The Labeling tab's <dataset>_labels.csv lands in the same folder, and whichever tab
    writes first creates it.
    """
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    flips = flips or {}
    fx, fy, fz = (bool(flips.get(key)) for _, key, _ in FLIPS)

    # The source header already carries the lab convention (0.54 isotropic spacing, an
    # RGB-color 4th axis), so it is reused rather than rebuilt.
    rescaled = rescale_channels(apply_flips(raw_data, fx, fy, fz), percentages)
    pal_path = os.path.join(out_dir, "NeuroPAL_corrected.nrrd")
    nrrd.write(pal_path, rescaled, header=nrrd.read_header(src_path))
    lines.append(f"NeuroPAL_corrected.nrrd {rescaled.shape} {rescaled.dtype}")

    if companion_paths is None:
        companion_paths = resolve_companion_paths(src_path)

    for name in COMPANION_VOLUMES:
        in_path = companion_paths[name]
        if not os.path.exists(in_path):
            lines.append(f"{os.path.basename(in_path)} not found, skipped")
            continue
        vol, hdr = nrrd.read(in_path)
        vol = apply_flips(vol, fx, fy, fz)  # dtype preserved: uint16 labels stay uint16
        out_name = f"{os.path.splitext(os.path.basename(in_path))[0]}_corrected.nrrd"
        nrrd.write(os.path.join(out_dir, out_name), vol, header=hdr)
        lines.append(f"{out_name} {vol.shape} {vol.dtype}")

    info_path = os.path.join(out_dir, "neuropal_info.txt")
    write_info_file(info_path, src_path, percentages, flips, source_code)
    lines.append("neuropal_info.txt")
    return out_dir, lines


# --------------------------------------------------------------------------------------
# Contrast curve
# --------------------------------------------------------------------------------------

MIN_POINT_GAP = 0.01         # keeps control points strictly ordered in x, for PCHIP
HANDLE_PICK_PX = 14          # how close the cursor must be to grab a control point


class ContrastCurve:
    """ITK-SNAP's curve-based contrast adjustment panel.

    Image intensity on x, index into the color map on y, the image histogram behind, and
    a grey ramp under the axis showing what the curve currently maps intensities to.

    All three control points drag freely in both x and y, as in ITK-SNAP. The only
    constraints are the ones that keep the curve a usable transfer function: points stay
    ordered in x, and their y values stay non-decreasing, so the mapping never folds back
    on itself. Points may share a y value, which is how the flat shoulder in ITK-SNAP's
    default view is made.

    Display only: nothing here reaches the saved volumes.
    """

    def __init__(self, parent, on_change):
        self.on_change = on_change
        self.points = [list(p) for p in DEFAULT_POINTS]
        self.ref = 1.0
        self.hist = None
        self.dragging = None
        self.selected = 0

        self.fig = Figure(figsize=(2.75, 2.0), layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.X)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    # -- state -------------------------------------------------------------------------

    def as_tuples(self):
        return [(px, py) for px, py in self.points]

    def set_histogram(self, values, ref):
        """Give the panel the intensities it should draw behind the curve."""
        self.ref = ref
        v = values[values > 0]
        if v.size:
            counts, edges = np.histogram(np.clip(v, 0.0, 1.0), bins=128, range=(0.0, 1.0))
            self.hist = (counts, edges)
        else:
            self.hist = None

    def reset(self):
        self.points = [list(p) for p in DEFAULT_POINTS]
        self.selected = 0
        self.draw()
        self.on_change()

    def selection_text(self):
        px, py = self.points[self.selected]
        return f"x  {px * self.ref:.0f}     y  {py:.3f}"

    # -- events ------------------------------------------------------------------------

    def _nearest_handle(self, event):
        """Which point the cursor is on, measured in pixels so both axes count equally."""
        best, best_d = None, HANDLE_PICK_PX
        for i, (px, py) in enumerate(self.points):
            sx, sy = self.ax.transData.transform((px, py))
            d = np.hypot(sx - event.x, sy - event.y)
            if d < best_d:
                best, best_d = i, d
        return best

    def on_press(self, event):
        if event.inaxes is not self.ax:
            return
        i = self._nearest_handle(event)
        if i is not None:
            self.dragging = i
            self.selected = i
            self.draw()

    def on_motion(self, event):
        if self.dragging is None or event.inaxes is not self.ax:
            return
        i = self.dragging
        # x stays between the neighbouring points; y stays between their y values, so the
        # curve keeps rising. Both are free otherwise, including for the end points.
        x_lo = 0.0 if i == 0 else self.points[i - 1][0] + MIN_POINT_GAP
        x_hi = 1.0 if i == len(self.points) - 1 else self.points[i + 1][0] - MIN_POINT_GAP
        y_lo = 0.0 if i == 0 else self.points[i - 1][1]
        y_hi = 1.0 if i == len(self.points) - 1 else self.points[i + 1][1]

        self.points[i][0] = float(np.clip(event.xdata, min(x_lo, x_hi), max(x_lo, x_hi)))
        self.points[i][1] = float(np.clip(event.ydata, y_lo, y_hi))

        self.draw()
        self.on_change()

    def on_release(self, event):
        self.dragging = None

    # -- drawing -----------------------------------------------------------------------

    def draw(self):
        self.ax.clear()
        strip_top, strip_bottom = -0.02, -0.10

        if self.hist is not None:
            counts, edges = self.hist
            centers = (edges[:-1] + edges[1:]) / 2.0
            scaled = np.log1p(counts) / max(np.log1p(counts).max(), 1e-8)
            self.ax.bar(centers, scaled, width=(edges[1] - edges[0]), color="#b9bede",
                        edgecolor="#8d93bf", linewidth=0.3, zorder=0)

        # the grey ramp under the axis: what the curve currently maps intensities to
        ramp = apply_transfer(np.linspace(0, 1, 256), self.as_tuples())
        self.ax.imshow(ramp[np.newaxis, :], extent=(0, 1, strip_bottom, strip_top),
                       aspect="auto", cmap="gray", vmin=0, vmax=1, zorder=1)

        xs = np.linspace(0.0, 1.0, 512)
        self.ax.plot(xs, apply_transfer(xs, self.as_tuples()),
                     color="#e02020", linewidth=2.0, zorder=2)

        for i, (px, py) in enumerate(self.points):
            self.ax.plot([px], [py], marker="o", markersize=10, zorder=3,
                         markerfacecolor="#ffe94d", markeredgecolor="black",
                         markeredgewidth=1.2 + 0.8 * (i == self.selected))

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(strip_bottom - 0.02, 1.06)
        # x ticks in raw intensity units, the numbers the volume actually holds
        ticks = np.linspace(0, 1, 5)
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([f"{t * self.ref:.0f}" for t in ticks], fontsize=7)
        self.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.ax.tick_params(axis="y", labelsize=7)
        self.ax.set_xlabel("Image Intensity", fontsize=8)
        self.ax.set_ylabel("Index into Color Map", fontsize=8)
        self.ax.grid(alpha=0.25, linewidth=0.5)
        self.canvas.draw_idle()


# --------------------------------------------------------------------------------------
# Slice scrollers
# --------------------------------------------------------------------------------------

SCROLL_COL_WIDTH = 26        # px reserved to the right of the views for the scrollers
SCROLL_PAGE = 10             # slices per page-jump, when the trough is clicked
REPEAT_DELAY_MS = 300        # hold an arrow this long before it starts repeating
REPEAT_INTERVAL_MS = 120     # and then step this often, matched to what a redraw costs
TOUCHPAD_UNITS_PER_SLICE = 10  # trackpad delta banked per slice, so a swipe is not a blur
# Tk 9 reports one detent as a flat +/-120 however fast the wheel is turned, where Tk 8.6
# on macOS passed the OS's accelerated count through (1 slice when eased, 13 when flicked).
# A single detent still steps one slice; only a fast run of them is scaled up.
WHEEL_ACCEL_MS = 100         # events closer together than this count as one flick
WHEEL_ACCEL_MAX = 6          # most slices a single accelerated detent may step


def wheel_delta(event):
    """One wheel event -> slices to step, sign included.

    Tk reports the wheel three different ways: Button-4/5 on X11, +/-120 steps on Windows,
    and small signed counts on macOS. Scrolling up steps to the earlier slice, the
    direction the thumb moves.
    """
    if getattr(event, "num", None) in (4, 5):
        return -1 if event.num == 4 else 1
    delta = getattr(event, "delta", 0)
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        delta /= 120.0
    return int(-delta) if abs(delta) >= 1 else (-1 if delta > 0 else 1)


def wheel_accel(dt_ms):
    """Slices per detent, given the gap since the previous wheel event.

    Only for wheels that report in fixed +/-120 steps, which carry no speed of their own.
    Eased scrolling (>= WHEEL_ACCEL_MS apart) stays at one slice per detent, so a single
    click still lands on the neighbouring slice.
    """
    if dt_ms >= WHEEL_ACCEL_MS:
        return 1
    return min(max(int(WHEEL_ACCEL_MS / max(dt_ms, 8.0)), 1), WHEEL_ACCEL_MAX)


def bind_wheel(widget, handler):
    """Route wheel and trackpad scrolling on `widget` to `handler(steps)`.

    Tk 9 routes a trackpad's precise scrolling to its own <TouchpadScroll> event, so on
    macOS a two-finger swipe never reaches <MouseWheel> there, though a wheel mouse still
    does. Tk 8.6 has no such event and sends both through <MouseWheel>; binding it is
    therefore attempted and allowed to fail.
    """
    last = [0.0]        # when the previous wheel event arrived, for the acceleration

    def on_wheel(event):
        steps = wheel_delta(event)
        if not steps:
            return "break"
        now = time.monotonic() * 1000.0
        # Quantised wheels only: a raw macOS delta is already accelerated by the OS.
        if abs(getattr(event, "delta", 0)) >= 120:
            steps *= wheel_accel(now - last[0])
        last[0] = now
        handler(steps)
        return "break"

    widget.bind("<MouseWheel>", on_wheel)      # macOS and Windows
    widget.bind("<Button-4>", on_wheel)        # X11
    widget.bind("<Button-5>", on_wheel)

    # A swipe arrives as a stream of small deltas, far more than there are slices to show,
    # so they are banked and spent a slice at a time.
    carry = [0.0]

    def on_touchpad(event):
        # %D packs both axes into one integer; Tk's own helper unpacks it.
        try:
            _, dy = (int(v) for v in widget.tk.splitlist(
                widget.tk.call("::tk::PreciseScrollDeltas", event.delta)))
        except (tk.TclError, ValueError):
            return "break"
        carry[0] -= dy                          # same sign convention as the wheel
        steps = int(carry[0] / TOUCHPAD_UNITS_PER_SLICE)
        if steps:
            carry[0] -= steps * TOUCHPAD_UNITS_PER_SLICE
            handler(steps)
        return "break"

    try:
        widget.bind("<TouchpadScroll>", on_touchpad)
    except tk.TclError:
        pass                                    # Tk 8.6: handled by <MouseWheel> above


class SliceScroller:
    """ITK-SNAP's per-panel slice scroller: an up arrow, a bar with a thumb, a down arrow.

    ttk's scrollbar carries no arrow buttons under the macOS theme, so the arrows are real
    buttons stacked above and below the bar. The thumb spans one slice out of the stack, so
    its size shows how deep the volume is and its position shows where in it you are.
    """

    def __init__(self, parent, on_index):
        self.on_index = on_index      # called with the slice index to move to
        self.index = 0
        self.n = 1
        self.repeat_job = None

        self.frame = ttk.Frame(parent)
        up = ttk.Button(self.frame, text="\u25b2", width=2, style="Slice.TButton")
        up.pack(side=tk.TOP, fill=tk.X)
        down = ttk.Button(self.frame, text="\u25bc", width=2, style="Slice.TButton")
        down.pack(side=tk.BOTTOM, fill=tk.X)
        self.bar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.on_bar)
        self.bar.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Held arrows repeat, as a scrollbar's do: one step on press, a pause, then a
        # steady stream until release. Driven from press/release rather than the button's
        # own command, so a hold is one gesture instead of a series of clicks.
        for button, delta in ((up, -1), (down, +1)):
            button.bind("<ButtonPress-1>", lambda e, d=delta: self.press(d))
            button.bind("<ButtonRelease-1>", lambda e: self.stop_repeat())
            button.bind("<Leave>", lambda e: self.stop_repeat())

        # and the wheel works anywhere over the strip, not just on the thumb
        for widget in (self.frame, self.bar, up, down):
            bind_wheel(widget, self.step)

    # -- auto-repeat -------------------------------------------------------------------

    def press(self, delta):
        self.step(delta)
        self.stop_repeat()
        self.repeat_job = self.frame.after(REPEAT_DELAY_MS, self.repeat, delta)

    def repeat(self, delta):
        self.step(delta)
        self.repeat_job = self.frame.after(REPEAT_INTERVAL_MS, self.repeat, delta)

    def stop_repeat(self):
        if self.repeat_job is not None:
            self.frame.after_cancel(self.repeat_job)
            self.repeat_job = None

    def set_state(self, index, n):
        """Point the thumb at slice `index` of `n`."""
        self.index, self.n = index, max(int(n), 1)
        if self.bar.winfo_exists():     # the window may be closing mid-redraw
            self.bar.set(self.index / self.n, (self.index + 1) / self.n)

    def step(self, delta):
        self.on_index(self.index + delta)

    def on_bar(self, *args):
        """Handle the three things a ttk scrollbar reports: drag, arrow, trough click."""
        if args[0] == "moveto":
            self.on_index(int(round(float(args[1]) * self.n)))
        elif args[0] == "scroll":
            unit = SCROLL_PAGE if args[2] == "pages" else 1
            self.on_index(self.index + int(args[1]) * unit)


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

class ScrollableFrame:
    """A fixed-width side column that scrolls when its sections outgrow a short screen.

    `inner` is what callers pack into. The wheel is bound across the whole column after it
    is filled, so scrolling works with the pointer over a slider or entry, not only over
    the gaps between them.
    """

    def __init__(self, parent, width):
        self.outer = ttk.Frame(parent, width=width)
        self.outer.pack_propagate(False)

        self.canvas = tk.Canvas(self.outer, width=width, highlightthickness=0,
                                borderwidth=0, takefocus=0)
        bar = ttk.Scrollbar(self.outer, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = ttk.Frame(self.canvas, padding=8)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self.window, width=e.width))

    def enable_wheel(self):
        """Bind the wheel over every widget in the column. Call once it is populated."""
        def scroll(steps):
            self.canvas.yview_scroll(steps, "units")

        def walk(widget):
            bind_wheel(widget, scroll)
            for child in widget.winfo_children():
                walk(child)
        bind_wheel(self.canvas, scroll)
        walk(self.inner)


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

# The side columns are laid out to fit a short screen without scrolling: a slightly
# smaller font than the platform default and tight padding throughout, so every section
# stays on screen at once.
#
# These are sizes in *points*, and Tk multiplies them by `tk scaling` to get pixels. That
# factor is not the same everywhere -- macOS Tk 8.6 reports ~1.0 with a 13pt default font,
# Tk 9 reports ~1.33 with a 10pt one -- so a hard-coded point size renders a third larger
# under one than the other. They are therefore derived from the platform's own default
# font at startup, in use_compact_style(), and these values are only the fallback.
PANEL_FONT = ("TkDefaultFont", 12)
HINT_FONT = ("TkDefaultFont", 11)
FIXED_FONT = ("TkFixedFont", 11)
BOX_PAD = 4               # inside a LabelFrame
BOX_GAP = 2               # between LabelFrames
ROW_GAP = 2               # between rows inside one

LEFT_PANEL_WIDTH = 292
RIGHT_PANEL_WIDTH = 322


def use_compact_style(root):
    """Shrink the default ttk font so the panels fit without needing to scroll.

    Sized relative to whatever this Tk calls normal, so the GUI looks the same whichever
    interpreter starts it -- a literal point size would not, because `tk scaling` differs
    between Tk versions.
    """
    global PANEL_FONT, HINT_FONT, FIXED_FONT
    base = int(tkfont.nametofont("TkDefaultFont").cget("size"))
    mono = int(tkfont.nametofont("TkFixedFont").cget("size"))
    PANEL_FONT = ("TkDefaultFont", max(base - 1, 1))
    HINT_FONT = ("TkDefaultFont", max(base - 2, 1))
    FIXED_FONT = ("TkFixedFont", max(mono, 1))

    style = ttk.Style(root)
    style.configure(".", font=PANEL_FONT)
    style.configure("TLabelframe.Label", font=PANEL_FONT)
    style.configure("Slice.TButton", padding=0)
    return style


class VolumeTab:
    """Shared machinery of both tabs: orientation, channels, overlays, curve, two views.

    Each tab keeps its own channel percentages, overlay opacities, curve and crosshair, so
    dialing one never moves the other. What they share is the loaded volume, the display
    orientation and the display white point, all of which live on the app.

    Subclasses supply the extra panels and set `slice_mode`, which decides whether the
    views are projections of the whole volume or one slice at a time. The overlays follow:
    a projection in a MIP tab, the matching slice in a slice tab.
    """

    slice_mode = False
    overlays_off = ()         # overlay keys a tab wants switched off on startup
    orientation_help = True   # show the notes under the orientation dropdown
    view_anchors = ()         # per-view anchor in its layout cell; empty leaves the
                              # matplotlib default, which centres the image

    def __init__(self, app, parent):
        self.app = app
        self.frame = ttk.Frame(parent)

        self.cursor = None        # [x, y, z] voxel the crosshair sits on
        self.rendered = None      # per-view RGB, reused while only the cursor moves
        self.axes = []
        self.scrollers = []

        self.pct_vars = {k: tk.DoubleVar(value=100.0) for k, _, _ in CHANNELS}
        self.entry_vars = {k: tk.StringVar(value="100") for k, _, _ in CHANNELS}
        self.visible = {k: tk.BooleanVar(value=True) for k, _, _ in CHANNELS}
        self.scales = {}
        # Master switch over the three channels: off blanks the NeuroPAL image without
        # disturbing which individual channels were selected, so they come back as they
        # were. Overlays are drawn on top either way.
        self.neuropal_on = tk.BooleanVar(value=True)
        self.selected_point_var = tk.StringVar(value="Selected control point:")

        self.overlay_on = {k: tk.BooleanVar(value=on and k not in self.overlays_off)
                           for k, _, _, _, _, on in OVERLAYS}
        self.overlay_pct = {k: tk.DoubleVar(value=d) for k, _, _, _, d, _ in OVERLAYS}
        self.overlay_entry = {k: tk.StringVar(value=f"{d:g}")
                              for k, _, _, _, d, _ in OVERLAYS}
        self.overlay_checks = {}
        self.keys_bound = False

        # The label table, shared machinery: Labeling writes it, Processing can display it.
        self.labels_path = None
        self.saved = {}                                   # roi id -> row
        self.show_labels = tk.BooleanVar(value=False)
        self.labels_path_var = tk.StringVar(value="")
        self.conf_filter_on = tk.BooleanVar(value=True)
        self.conf_threshold = tk.DoubleVar(value=CONFIDENCE_THRESHOLD_DEFAULT)
        self.conf_threshold_label = tk.StringVar(
            value=str(CONFIDENCE_THRESHOLD_DEFAULT))
        # Checked hides the left column. Two checkbuttons share this variable: one at the
        # top of the column, and a stand-in at the top-left of the images that only appears
        # once the column is gone, so the switch is never hidden along with it.
        self.hide_left = tk.BooleanVar(value=False)

        self.build()

    # -- layout ------------------------------------------------------------------------

    def build(self):
        self.build_header(self.frame)

        body = ttk.Frame(self.frame)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.body = body
        self.left = ttk.Frame(body, width=LEFT_PANEL_WIDTH, padding=(6, 4))
        self.left.pack(side=tk.LEFT, fill=tk.Y)
        self.left.pack_propagate(False)
        ttk.Checkbutton(self.left, text="Hide left panel display",
                        variable=self.hide_left,
                        command=self.toggle_left_panel).pack(anchor=tk.W, pady=(0, 2))
        self.build_controls(self.left)

        self.right = ttk.Frame(body, width=RIGHT_PANEL_WIDTH, padding=(4, 4, 6, 4))
        self.right.pack(side=tk.RIGHT, fill=tk.Y)
        self.right.pack_propagate(False)
        self.build_contrast(self.right)
        self.build_right_extras(self.right)

        self.build_views(body)

    def toggle_left_panel(self):
        """Collapse the left column, or bring it back into its original slot."""
        if self.hide_left.get():
            self.left.pack_forget()
            self.hide_bar.pack(side=tk.TOP, fill=tk.X, before=self.viewport)
        else:
            self.hide_bar.pack_forget()
            self.left.pack(side=tk.LEFT, fill=tk.Y, before=self.plot)

    def build_header(self, parent):
        """Anything above the panels. Only the Processing tab uses this."""

    def build_right_extras(self, parent):
        """Anything under the contrast curve."""

    def build_controls(self, parent):
        """The left column, in the order both tabs share."""
        self.build_orientation(parent)
        self.build_channels(parent)
        self.build_cursor(parent)
        self.build_overlays(parent)

    def build_orientation(self, parent):
        """Display orientation, as an anatomical code rather than three tick boxes.

        The code names where each axis points as its index grows -- "LPS" is +x to Left,
        +y to Posterior, +z to Superior. Picking a different one turns the view round; it
        never touches the data, only how Processing then writes it out.
        """
        box = ttk.LabelFrame(parent, text="Display orientation", padding=BOX_PAD)
        box.pack(fill=tk.X, pady=BOX_GAP)

        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Show as ").pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=self.app.orientation_var,
                             state="readonly", width=14)
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        combo.bind("<<ComboboxSelected>>", lambda e: self.app.on_orientation_selected())
        self.app.orientation_boxes.append(combo)

        # The notes belong with the tab that decides the orientation and writes it out.
        # Labeling gets the dropdown alone, so the view can still be turned round.
        if not self.orientation_help:
            return

        ttk.Label(box, textvariable=self.app.source_note_var, foreground="#555",
                  font=HINT_FONT).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(box, text="Only rotations of the source orientation are offered; "
                            "reflections would swap left and right.",
                  justify=tk.LEFT, foreground="#555", font=HINT_FONT,
                  wraplength=LEFT_PANEL_WIDTH - 40).pack(anchor=tk.W, padx=3, pady=(2, 1))
        ttk.Label(box, text="Orient the lateral view with the ventral nerve cord at the "
                            "bottom; left-side neurons then appear at the bottom of the "
                            "coronal view (smaller z).",
                  justify=tk.LEFT, foreground="#555", font=HINT_FONT,
                  wraplength=LEFT_PANEL_WIDTH - 40).pack(anchor=tk.W, padx=3, pady=(4, 1))

    def build_channels(self, parent):
        box = ttk.LabelFrame(parent, text="Channel brightness (% = 100 / factor)",
                             padding=BOX_PAD)
        box.pack(fill=tk.X, pady=BOX_GAP)

        ttk.Checkbutton(box, text="NeuroPAL image", variable=self.neuropal_on,
                        command=self.refresh).pack(anchor=tk.W)
        ttk.Separator(box).pack(fill=tk.X, pady=(4, 0))

        for key, _, _ in CHANNELS:
            row = ttk.Frame(box)
            row.pack(fill=tk.X, pady=(ROW_GAP, 0))
            ttk.Checkbutton(row, text=CHANNEL_SHORT[key], width=13,
                            variable=self.visible[key],
                            command=self.refresh).pack(side=tk.LEFT)

            entry = ttk.Entry(row, textvariable=self.entry_vars[key], width=5)
            entry.pack(side=tk.RIGHT)
            entry.bind("<Return>", lambda e, k=key: self.on_entry(k))
            entry.bind("<FocusOut>", lambda e, k=key: self.on_entry(k))

            scale = ttk.Scale(row, from_=0.0, to=SLIDER_MAX, variable=self.pct_vars[key],
                              command=lambda v, k=key: self.on_slider(k))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            self.scales[key] = scale

        ttk.Button(box, text="Reset all to 100%",
                   command=self.on_reset).pack(fill=tk.X, pady=(4, 0))

    def build_cursor(self, parent):
        """Whatever the tab shows about the crosshair."""

    def build_label_display(self, parent):
        """The `Show label on image` switch and the confidence threshold beside it."""
        ttk.Checkbutton(parent, text="Show label on image", variable=self.show_labels,
                        command=self.refresh).pack(anchor=tk.W)

        lo, hi = CONFIDENCE_THRESHOLD_RANGE
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(ROW_GAP, 0))
        # The heading is the switch: unticked, the threshold is ignored entirely, which is
        # what a table with no confidence column recorded needs.
        ttk.Checkbutton(row, text="Confidence threshold", variable=self.conf_filter_on,
                        command=self.on_conf_filter).pack(side=tk.LEFT)
        self.conf_readout = ttk.Label(row, textvariable=self.conf_threshold_label,
                                      width=2, anchor=tk.E)
        self.conf_readout.pack(side=tk.RIGHT)
        self.conf_scale = ttk.Scale(row, from_=lo, to=hi, variable=self.conf_threshold,
                                    command=lambda v: self.on_conf_threshold())
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Label(parent, text=f"(>= threshold), {lo} to {hi}; untick to show every name",
                  foreground="#555", font=HINT_FONT,
                  wraplength=RIGHT_PANEL_WIDTH - 30).pack(anchor=tk.W)
        ttk.Label(parent, text=f'press "{LABEL_HOTKEY}" to toggle the names',
                  foreground="#555", font=HINT_FONT).pack(anchor=tk.W)

    def confidence_threshold(self):
        """The slider's value as a whole number.

        The variable itself is left continuous -- snapping it inside the scale's own
        callback would re-enter that callback -- so it is rounded here and at the readout.
        """
        return int(round(float(self.conf_threshold.get())))

    def on_conf_threshold(self):
        self.conf_threshold_label.set(str(self.confidence_threshold()))
        self.refresh()

    def on_conf_filter(self):
        """Grey the slider out when the threshold is switched off, then redraw."""
        state = ["!disabled"] if self.conf_filter_on.get() else ["disabled"]
        self.conf_scale.state(state)
        self.conf_readout.state(state)
        self.refresh()

    def build_labels_path_row(self, parent):
        """`Path: [box] Browse` on one line, showing the tail of the path."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(ROW_GAP, 0))
        ttk.Label(row, text="Path ").pack(side=tk.LEFT)
        ttk.Button(row, text="Browse", width=7,
                   command=self.on_browse_labels).pack(side=tk.RIGHT)
        entry = ttk.Entry(row, textvariable=self.labels_path_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        entry.bind("<Return>", lambda e: self.use_typed_labels_path())
        self.app.show_path_tail(entry, self.labels_path_var)
        return row

    def on_browse_labels(self):
        """Point this tab at a different label table."""
        path = filedialog.askopenfilename(
            title="Select a label table",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.set_labels_path(path)

    def use_typed_labels_path(self):
        self.set_labels_path(self.labels_path_var.get().strip())

    def set_labels_path(self, path, propagate=True):
        """Point at a label table. Both tabs follow, so they never read different files.

        A browsed table also becomes the app's override, so it survives the next Load
        instead of quietly snapping back to whatever the launcher named.
        """
        self.labels_path = path or None
        self.labels_path_var.set(path or "")
        self.app.labels_override = path or None
        self.reload_labels(force=True)
        if propagate:
            for tab in self.app.tabs:
                if tab is not self:
                    tab.set_labels_path(path, propagate=False)

    def reload_labels(self, force=False):
        """Re-read the table from disk; redraw if what it holds changed.

        Called when a tab is shown, so names labelled in one tab appear in the other
        without reloading the volume.
        """
        if not self.labels_path:
            return
        rows = read_labels(self.labels_path)
        if force or rows != self.saved:
            self.saved = rows
            self.rendered = None        # names are drawn during the redraw
            if hasattr(self, "refresh_find_choices"):
                self.refresh_find_choices()
            self.refresh()

    def build_overlays(self, parent):
        kind = "slice" if self.slice_mode else "maximum intensity projection"
        box = ttk.LabelFrame(parent, text=f"Overlays ({kind}, display only)",
                             padding=BOX_PAD)
        box.pack(fill=tk.X, pady=BOX_GAP)
        for i, (key, _, label, hotkey, _, _) in enumerate(OVERLAYS):
            if i:
                ttk.Separator(box).pack(fill=tk.X, pady=4)
            self.build_overlay_row(box, key, label, hotkey)

    def build_overlay_row(self, parent, key, label, hotkey):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(ROW_GAP, 0))
        check = ttk.Checkbutton(row, text=label, width=13, variable=self.overlay_on[key],
                                command=self.refresh)
        check.pack(side=tk.LEFT)
        self.overlay_checks[key] = check

        entry = ttk.Entry(row, textvariable=self.overlay_entry[key], width=5)
        entry.pack(side=tk.RIGHT)
        entry.bind("<Return>", lambda e, k=key: self.on_overlay_entry(k))
        entry.bind("<FocusOut>", lambda e, k=key: self.on_overlay_entry(k))

        ttk.Scale(row, from_=0.0, to=100.0, variable=self.overlay_pct[key],
                  command=lambda v, k=key: self.on_overlay_slider(k)).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Label(parent, text=f'opacity %, press "{hotkey}" to toggle',
                  foreground="#555", font=HINT_FONT).pack(anchor=tk.W)

    def build_contrast(self, parent):
        """The contrast curve, at the top of the right column. Display only, never saved."""
        box = ttk.LabelFrame(parent, text="Curve-Based Contrast Adjustment (display only)",
                             padding=BOX_PAD)
        box.pack(side=tk.TOP, fill=tk.X)

        self.curve = ContrastCurve(box, on_change=self.on_curve)
        ttk.Label(box, textvariable=self.selected_point_var, justify=tk.LEFT,
                  font=HINT_FONT).pack(anchor=tk.W, pady=(3, 0))
        ttk.Button(box, text="Reset curve",
                   command=self.curve.reset).pack(fill=tk.X, pady=(3, 0))

    def build_views(self, parent):
        plot = ttk.Frame(parent)
        plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot = plot

        viewport = ttk.Frame(plot)
        self.viewport = viewport        # packed below, after the toolbar has its height

        # The stand-in switch, parked at the top left of the images. It stays unpacked
        # while the column is open, so it costs no height until it is actually needed.
        self.hide_bar = ttk.Frame(plot)
        ttk.Checkbutton(self.hide_bar, text="Hide left panel display",
                        variable=self.hide_left,
                        command=self.toggle_left_panel).pack(side=tk.LEFT, padx=2)

        # Lateral on top, coronal directly below and sharing its x axis, so the same
        # head-to-tail position lines up vertically between the two. The layout padding is
        # tightened from the default so the two panels sit close rather than floating in
        # white space -- the images are width-limited, so any slack shows up between them.
        self.fig = Figure(figsize=(9, 6), layout="constrained")
        self.fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.0, wspace=0.0)
        self.axes = self.fig.subplots(len(VIEWS), 1, sharex=True)
        self.anchor_views()

        if self.slice_mode:
            self.scroll_col = ttk.Frame(viewport, width=SCROLL_COL_WIDTH)
            self.scroll_col.pack(side=tk.RIGHT, fill=tk.Y)
            self.scroll_col.pack_propagate(False)
            self.scrollers = [
                SliceScroller(self.scroll_col, lambda i, v=v: self.on_scroller(v, i))
                for v in range(len(VIEWS))]

        self.canvas = FigureCanvasTkAgg(self.fig, master=viewport)
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Pack the toolbar before the viewport so it reserves its full height first. The
        # viewport expands, and packing it first left the toolbar 11px of the 50 it needs
        # in the Processing tab, where the loader row makes the plot area shorter.
        toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_drag)
        if self.slice_mode:
            # Tk's own wheel event rather than matplotlib's scroll_event: it fires on
            # hover without the figure having been clicked first, and it carries trackpad
            # scrolling on macOS.
            bind_wheel(widget, self.on_wheel_over_image)
            # Each scroller is parked against its own panel, so it has to follow the axes
            # whenever the layout is recomputed -- on resize, or on a new volume.
            self.canvas.mpl_connect("draw_event", lambda e: self.place_scrollers())
            # so the wheel reaches the figure without having to click it first
            widget.bind("<Enter>", lambda e: widget.focus_set())

    def place_scrollers(self):
        """Line each scroller up with the vertical extent of the panel it drives."""
        for scroller, ax in zip(self.scrollers, self.axes):
            box = ax.get_position()
            # matplotlib measures y from the bottom, Tk from the top
            scroller.frame.place(relx=0, rely=1 - box.y1,
                                 relwidth=1, relheight=box.height)

    def make_view_axes(self):
        """Rebuild the view panels with heights matching the data, so both share a width.

        Each panel draws at true pixel aspect, so without this the taller lateral view
        would be squeezed narrower than the coronal one sitting under it.
        """
        # displayed rows: y for the lateral view (axis 2), z for the coronal (axis 1)
        ratios = [self.app.raw.shape[3 - axis] for axis, _ in VIEWS]
        self.fig.clear()
        self.axes = self.fig.subplots(len(VIEWS), 1, sharex=True,
                                      gridspec_kw={"height_ratios": ratios})
        self.anchor_views()

    def anchor_views(self):
        """Place each panel inside its layout cell, per `view_anchors`.

        Each image is drawn at true aspect, so a window taller than the images need leaves
        slack in every cell. Matplotlib's default centres the image in its cell, which is
        what both tabs use unless a tab asks otherwise; anchoring both panels towards each
        other was tried and reverted, because closing the gap that far pushed an awkward
        band above the images instead.
        """
        for i, ax in enumerate(self.axes):
            if self.view_anchors:
                ax.set_anchor(self.view_anchors[i])
            ax.axis("off")

    # -- state -------------------------------------------------------------------------

    def percentages(self):
        return {k: float(self.pct_vars[k].get()) for k, _, _ in CHANNELS}

    def visibility(self):
        """Which channels get drawn: each one's own box, gated by the master switch."""
        master = bool(self.neuropal_on.get())
        return {k: master and bool(self.visible[k].get()) for k, _, _ in CHANNELS}

    def opacity(self, key):
        return float(self.overlay_pct[key].get()) / 100.0

    def overlay_visible(self, key, volume_name):
        return (bool(self.overlay_on[key].get())
                and self.app.companion_data.get(volume_name) is not None
                and self.opacity(key) > 0)

    def on_volume_loaded(self):
        """A new volume was loaded by the app: reset what depends on its shape."""
        ref = self.app.ref
        self.curve.set_histogram(self.app.raw[..., :3].max(axis=2).ravel() / ref, ref)
        self.curve.draw()
        self.selected_point_var.set("Selected control point:   "
                                    + self.curve.selection_text())
        self.cursor = [d // 2 for d in self.app.raw.shape[:3]]
        self.labels_path = self.app.labels_file()
        self.labels_path_var.set(self.labels_path)
        self.saved = read_labels(self.labels_path)
        for key, volume_name, _, _, _, _ in OVERLAYS:
            available = self.app.companion_data.get(volume_name) is not None
            self.overlay_checks[key].state(["!disabled"] if available else ["disabled"])
        self.make_view_axes()
        self.invalidate()

    def invalidate(self):
        """The pixels changed underneath us (a new volume, or a flip), so re-render."""
        self.rendered = None
        self.refresh()

    def refresh_if_stale(self):
        """Redraw only if something changed while this tab was hidden.

        Switching tabs used to redraw unconditionally, which cost a full matplotlib draw
        every time even when the panel was already showing the right thing.
        """
        if self.rendered is None:
            self.refresh()

    # -- callbacks ---------------------------------------------------------------------

    def on_slider(self, key):
        self.entry_vars[key].set(f"{self.pct_vars[key].get():.4g}")
        self.refresh()

    def on_entry(self, key):
        """Accept a typed percentage, clamped to the slider's range."""
        try:
            val = float(self.entry_vars[key].get())
        except ValueError:
            self.entry_vars[key].set(f"{self.pct_vars[key].get():.4g}")
            return
        val = min(max(0.0, val), SLIDER_MAX)
        self.pct_vars[key].set(val)
        self.entry_vars[key].set(f"{val:.4g}")
        self.refresh()

    def on_reset(self):
        self.neuropal_on.set(True)
        for key, _, _ in CHANNELS:
            self.pct_vars[key].set(100.0)
            self.entry_vars[key].set("100")
            self.visible[key].set(True)
        self.refresh()

    def on_overlay_slider(self, key):
        self.overlay_entry[key].set(f"{self.overlay_pct[key].get():.4g}")
        self.refresh()

    def on_overlay_entry(self, key):
        try:
            val = float(self.overlay_entry[key].get())
        except ValueError:
            self.overlay_entry[key].set(f"{self.overlay_pct[key].get():.4g}")
            return
        val = min(max(0.0, val), 100.0)
        self.overlay_pct[key].set(val)
        self.overlay_entry[key].set(f"{val:.4g}")
        self.refresh()

    def on_curve(self):
        """The curve changed: update the control-point readout and redraw the views."""
        self.selected_point_var.set("Selected control point:   "
                                    + self.curve.selection_text())
        self.refresh()

    # -- hotkeys -----------------------------------------------------------------------

    def claim_keys(self):
        """Take the 'a' and 's' toggles while this tab is the visible one."""
        if self.keys_bound:
            return
        for key, _, _, hotkey, _, _ in OVERLAYS:
            self.app.root.bind_all(f"<KeyPress-{hotkey}>",
                                   lambda e, k=key: self.on_hotkey(k))
        self.app.root.bind_all(f"<KeyPress-{LABEL_HOTKEY}>",
                               lambda e: self.on_label_hotkey())
        self.keys_bound = True

    def release_keys(self):
        if not self.keys_bound:
            return
        for _, _, _, hotkey, _, _ in OVERLAYS:
            self.app.root.unbind_all(f"<KeyPress-{hotkey}>")
        self.app.root.unbind_all(f"<KeyPress-{LABEL_HOTKEY}>")
        self.keys_bound = False

    def hotkey_allowed(self):
        """False while an editable text box has focus -- the keystroke belongs to it.

        A readonly box still lets a hotkey through: nothing can be typed into it, so the
        key cannot have been meant for it.
        """
        focused = self.app.root.focus_get()
        if isinstance(focused, (ttk.Entry, tk.Entry)):
            try:
                return str(focused.cget("state")) == "readonly"
            except tk.TclError:
                return False
        return True

    def on_hotkey(self, key):
        """Toggle an overlay, unless the keystroke was meant for a text box."""
        if not self.hotkey_allowed():
            return
        self.overlay_on[key].set(not self.overlay_on[key].get())
        self.refresh()

    def on_label_hotkey(self):
        """Toggle the neuron names, same guard as the overlay hotkeys."""
        if not self.hotkey_allowed():
            return
        self.show_labels.set(not self.show_labels.get())
        self.refresh()

    # -- crosshair and slices ----------------------------------------------------------

    def on_canvas_click(self, event):
        if event.button == 1:
            self.move_cursor(event)

    def on_canvas_drag(self, event):
        """Scrub the crosshair while the left button is held, as ITK-SNAP does."""
        if event.button == 1:
            self.move_cursor(event)

    def on_wheel_over_image(self, steps):
        """Step through slices with the wheel, in whichever view is under the pointer.

        The slice index of a view *is* the cursor coordinate along its axis, so scrolling
        the lateral view moves the cursor in z, which is also the horizontal crosshair
        line the coronal view draws. The two stay in step for free.
        """
        if self.app.data is None or not self.slice_mode:
            return
        view = self.view_under_pointer()
        if view is not None:
            self.set_slice(view, self.cursor[VIEWS[view][0]] + steps)

    def view_under_pointer(self):
        """Which panel the mouse is over, from the pointer position on the canvas."""
        widget = self.canvas.get_tk_widget()
        x = widget.winfo_pointerx() - widget.winfo_rootx()
        y = widget.winfo_pointery() - widget.winfo_rooty()
        # matplotlib measures display coordinates from the bottom
        point = (x, widget.winfo_height() - y)
        for view, ax in enumerate(self.axes):
            if ax.bbox.contains(*point):
                return view
        return None

    def on_scroller(self, view, index):
        """A scroller's arrow, thumb or trough moved."""
        self.set_slice(view, index)

    def set_slice(self, view, index):
        """Move one view to a slice, clamped to the volume."""
        if self.app.data is None or self.cursor is None:
            return
        axis = VIEWS[view][0]
        self.cursor[axis] = int(np.clip(index, 0, self.app.data.shape[axis] - 1))
        self.refresh()

    def move_cursor(self, event):
        """Place the 3D cursor from a click in whichever view was hit.

        Each panel sets the two coordinates it shows: the lateral view sets x and y, the
        coronal sets x and z. x is common to both, which is what links them -- clicking a
        neuron in one view slides the other's vertical line onto the same position along
        the worm. In slice mode it does more than that: the y a lateral click sets is the
        coronal view's own slice index, so the coronal image jumps to that slice.
        """
        if self.app.data is None or event.xdata is None or event.ydata is None:
            return
        for ax, (axis, _) in zip(self.axes, VIEWS):
            if event.inaxes is not ax:
                continue
            vaxis = 3 - axis          # lateral shows y, coronal shows z
            self.cursor[0] = int(np.clip(round(event.xdata), 0,
                                         self.app.data.shape[0] - 1))
            self.cursor[vaxis] = int(np.clip(round(event.ydata), 0,
                                             self.app.data.shape[vaxis] - 1))
            # In slice mode the cursor picks which slices are shown, so the pixels really
            # do change and the cache cannot be reused.
            self.refresh(recompute=self.slice_mode)
            return

    # -- drawing -----------------------------------------------------------------------

    def view_title(self, axis, name):
        """Panel caption: which projection, or which slice of how many, ITK-SNAP style."""
        if not self.slice_mode:
            return f"{name}  (MIP over {AXIS_LETTER[axis]})"
        return (f"{name}  ({AXIS_LETTER[axis]} {self.cursor[axis] + 1} "
                f"of {self.app.data.shape[axis]})")

    def render_frames(self, pcts, vis, points):
        if self.slice_mode:
            return [composite_slice(self.app.data, pcts, vis, self.app.ref, points=points,
                                    axis=axis, index=self.cursor[axis])
                    for axis, _ in VIEWS]
        return [composite_mip(self.app.data, pcts, vis, self.app.ref, points=points,
                              axis=axis)
                for axis, _ in VIEWS]

    def companion_plane(self, name, axis):
        """The companion volume reduced the same way the base image is for this tab."""
        vol = self.app.companion_data.get(name)
        if vol is None:
            return None
        if self.slice_mode:
            return np.take(vol, self.cursor[axis], axis=axis)
        return vol.max(axis=axis)

    def overlay_specs(self, axis):
        """all_red and the segmentation for this view, as {key: imshow kwargs}.

        all_red is blended per pixel rather than as a flat wash: its alpha is its own
        intensity times the opacity, so bright structure shows through as grey and empty
        background leaves the NeuroPAL image untouched. The segmentation is masked at 0,
        so only labelled voxels are painted, and is scaled against the whole label range so
        an ROI keeps its colour from slice to slice.
        """
        specs = {}

        if self.overlay_visible("red", "all_red"):
            plane = self.companion_plane("all_red", axis)
            norm = np.clip(plane.astype(np.float32) / self.app.all_red_ref, 0.0, 1.0)
            specs["red"] = dict(X=norm.T, cmap="gray", vmin=0.0, vmax=1.0,
                                alpha=(norm * self.opacity("red")).T)

        if self.overlay_visible("seg", "neuron_rois"):
            plane = self.companion_plane("neuron_rois", axis)
            specs["seg"] = dict(X=np.ma.masked_where(plane == 0, plane).T,
                                cmap=SEGMENTATION_CMAP, vmin=1, vmax=self.app.roi_max,
                                alpha=self.opacity("seg"))
        return specs

    def annotate(self, ax, axis):
        """Write each saved neuron's name over its ROI.

        In slice mode a name is drawn only on the slices where its ROI actually has
        voxels, at the ROI's centroid *within that slice*, so it follows the blob as you
        scroll. In MIP mode the projection collapses the whole stack, so every labelled
        ROI is drawn, at its 3D centroid projected onto the panel -- the names are
        projected in the same axis as the image under them.
        """
        rois = self.app.companion_data.get("neuron_rois")
        if not self.show_labels.get() or rois is None or not self.saved:
            return

        filtering = bool(self.conf_filter_on.get())
        threshold = self.confidence_threshold()
        for roi_id, row in self.saved.items():
            if not (row["Neuron Class"] or "").strip():
                continue            # a row kept for an ROI that has no name yet
            if filtering:
                # A row with no confidence recorded cannot clear a threshold, so it only
                # shows with the filter off -- which is the point of the switch.
                try:
                    confidence = int(row["Confidence"])
                except (TypeError, ValueError):
                    confidence = None
                if confidence is None or confidence < threshold:
                    continue        # below what the threshold asks to see
            if self.slice_mode:
                plane = np.take(rois, self.cursor[axis], axis=axis)
                here = np.argwhere(plane == roi_id)
                if not len(here):
                    continue            # this ROI is not in the slice on screen
                cx, cv = here.mean(axis=0)
            else:
                centre = self.app.roi_centroid(roi_id)
                if centre is None:
                    continue            # labelled ROI is not in this volume
                cx, cv = centre[0], centre[3 - axis]
            ax.text(cx, cv, row["Neuron Class"], color=LABEL_TEXT_COLOR,
                    fontsize=LABEL_FONT_SIZE, fontweight="bold",
                    ha="center", va="center",
                    path_effects=[patheffects.withStroke(linewidth=1,
                                                         foreground="black")])

    def header_line(self, pcts, shown):
        return ("  ".join(f"{k} {pcts[k]:.4g}%" for k in shown))

    def refresh(self, recompute=True):
        """Redraw both views. `recompute=False` reuses the frames already rendered, which
        is what keeps dragging the crosshair smooth in MIP mode -- there the cursor does
        not change any pixel, only the lines drawn over them."""
        if self.app.data is None or self.cursor is None:
            return
        pcts = self.percentages()
        vis = self.visibility()
        points = self.curve.as_tuples()
        shown = [k for k, _, _ in CHANNELS if vis[k]]

        if recompute or self.rendered is None:
            # Both views come off the same volume, and go through the same brightness
            # percentages, white point and curve, so their channel balance is identical
            # and the two are directly comparable.
            self.rendered = self.render_frames(pcts, vis, points)

        for ax, (axis, name), rgb in zip(self.axes, VIEWS, self.rendered):
            ax.clear()
            # (X, Y, C) -> (Y, X, C), so x runs along the display's horizontal axis.
            ax.imshow(np.transpose(rgb, (1, 0, 2)), origin=VIEW_ORIGIN[axis])
            for spec in self.overlay_specs(axis).values():
                kwargs = dict(spec)
                ax.imshow(kwargs.pop("X"), origin=VIEW_ORIGIN[axis],
                          interpolation="nearest", **kwargs)
            ax.axis("off")
            ax.set_title(self.view_title(axis, name), fontsize=9)

            # x is shared by both panels; the horizontal line is each panel's own in-plane
            # axis, y for the lateral view and z for the coronal.
            ax.axvline(self.cursor[0], color=CURSOR_COLOR, linestyle="--",
                       linewidth=0.9, alpha=0.9)
            ax.axhline(self.cursor[3 - axis], color=CURSOR_COLOR, linestyle="--",
                       linewidth=0.9, alpha=0.9)
            self.annotate(ax, axis)

        axis0, name0 = VIEWS[0]
        self.axes[0].set_title(
            f"{os.path.basename(self.app.src_path)}  |  {self.header_line(pcts, shown)}"
            f"\n{self.view_title(axis0, name0)}", fontsize=9)

        for scroller, (axis, _) in zip(self.scrollers, VIEWS):
            scroller.set_state(self.cursor[axis], self.app.data.shape[axis])

        self.after_draw(pcts)
        self.canvas.draw_idle()

    def after_draw(self, pcts):
        """Hook for whatever readout the tab keeps beside the views."""


class ProcessingTab(VolumeTab):
    """Dial the three channels against the projections and write the corrected volumes.

    This is the only tab that can write a volume.
    """

    slice_mode = False
    # the segmentation is off to start with here: this tab is for judging channel balance,
    # and a projection of every ROI covers exactly the neurons being judged
    overlays_off = ("seg",)
    # the coronal panel rides at the top of its cell, lifting it about half a centimetre
    # off the bottom of the figure; the lateral one stays centred
    view_anchors = ("C", "N")

    def __init__(self, app, parent):
        self.cursor_var = tk.StringVar(value="click either view to place the crosshair")
        super().__init__(app, parent)

    def build_header(self, parent):
        """The three input paths side by side on one line, with Load underneath."""
        top = ttk.Frame(parent, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        row = ttk.Frame(top)
        row.pack(fill=tk.X)
        inputs = [("NeuroPAL.nrrd", self.app.path_var),
                  ("all_red.nrrd", self.app.companion_vars["all_red"]),
                  ("neuron_rois.nrrd", self.app.companion_vars["neuron_rois"])]
        for label, var in inputs:
            group = ttk.Frame(row)
            group.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            ttk.Label(group, text=f"{label}:").pack(side=tk.LEFT)
            entry = ttk.Entry(group, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
            ttk.Button(group, text="Browse", width=7,
                       command=lambda v=var, l=label: self.app.browse(v, l)).pack(
                           side=tk.LEFT)
            # There is only room for part of a path, so show its tail: the file name and
            # the folder it sits in say far more than the start of an absolute path.
            self.app.show_path_tail(entry, var)

        ttk.Button(top, text="Load", command=self.app.load).pack(side=tk.TOP, pady=(5, 0))

    def build_cursor(self, parent):
        box = ttk.LabelFrame(parent, text="Cursor", padding=BOX_PAD)
        box.pack(fill=tk.X, pady=BOX_GAP)
        ttk.Label(box, textvariable=self.cursor_var, justify=tk.LEFT,
                  font=FIXED_FONT).pack(anchor=tk.W)

    def build_right_extras(self, parent):
        box = ttk.LabelFrame(parent, text="Output", padding=BOX_PAD)
        box.pack(side=tk.TOP, fill=tk.X, pady=(BOX_GAP * 2, 0))
        ttk.Button(box, text="Save corrected NRRDs + info",
                   command=self.on_save_corrected).pack(fill=tk.X)
        ttk.Label(box, textvariable=self.app.output_var, justify=tk.LEFT,
                  foreground="#555", font=HINT_FONT,
                  wraplength=RIGHT_PANEL_WIDTH - 30).pack(anchor=tk.W)

        labels = ttk.LabelFrame(parent, text="Labeled Neurons (Optional)", padding=BOX_PAD)
        labels.pack(side=tk.TOP, fill=tk.X, pady=(BOX_GAP * 2, 0))
        self.build_label_display(labels)
        ttk.Label(labels, text="names are projected in z, like the image",
                  foreground="#555", font=HINT_FONT).pack(anchor=tk.W)
        self.build_labels_path_row(labels)

    def after_draw(self, pcts):
        self.cursor_var.set(self.cursor_text(pcts))

    def cursor_text(self, pcts):
        """Voxel under the crosshair: raw values, and what the save would write.

        The views are projections, so a bright spot in one does not fix the third
        coordinate -- click the same neuron in both panels to land the cursor on it.
        """
        x, y, z = self.cursor
        raw = [float(self.app.data[x, y, z, idx]) for _, _, idx in CHANNELS]
        out = []
        for (key, _, idx), v in zip(CHANNELS, raw):
            factor = pct_to_factor(pcts[key])
            out.append(0.0 if factor == float("inf") else v / factor)
        return (f"x {x}   y {y}   z {z}\n"
                + "raw        " + "  ".join(f"{k} {v:>7.1f}"
                                            for (k, _, _), v in zip(CHANNELS, raw))
                + "\ncorrected  " + "  ".join(f"{k} {v:>7.1f}"
                                              for (k, _, _), v in zip(CHANNELS, out)))

    def on_save_corrected(self):
        if self.app.src_path is None:
            messagebox.showwarning("Nothing loaded", "Load a NeuroPAL .nrrd first.")
            return
        pcts = self.percentages()
        try:
            # the pristine volume, not the flipped one on screen: the output keeps the
            # source's coordinates so it stays aligned with the inputs
            out_dir, written = save_corrected(
                self.app.src_path, self.app.raw, pcts, self.app.output_dir(),
                flips=self.app.flip_state(), source_code=self.app.source_code,
                companion_paths=self.app.companion_paths)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        summary = "\n".join(
            f"{label}: {pcts[k]:.4g}%, division factor "
            f"{'inf (channel off)' if pct_to_factor(pcts[k]) == float('inf') else f'{pct_to_factor(pcts[k]):.4g}'}"
            for k, label, _ in CHANNELS)
        messagebox.showinfo(
            "Saved", f"{out_dir}\n\n" + "\n".join(written) + f"\n\n{summary}")
        self.app.status_var.set(f"Saved {out_dir}")


class LabelingTab(VolumeTab):
    """Scroll through slices to find a neuron and record its identity.

    Read-only for volumes by construction: the panel offers no save of image data, and the
    tab holds no code that writes one. It writes only the label table.
    """

    slice_mode = True
    orientation_help = False  # the dropdown alone; the explanation lives in Processing

    def __init__(self, app, parent):
        self.coord_var = tk.StringVar(value="(load a volume in the Processing tab)")
        self.actual_var = tk.StringVar(value="-")
        self.rgb_var = tk.StringVar(value="-")
        self.label_var = tk.StringVar(value="-")
        self.find_var = tk.StringVar(value="")
        self.find_box = None

        # the entry boxes that read and write the table (the table itself is on the base)
        self.shown_roi = None     # roi the form is currently showing, to spot a change
        self.entry_id = tk.StringVar(value="")
        self.entry_pos = tk.StringVar(value="")
        self.entry_neuron = tk.StringVar(value="")
        self.entry_conf = tk.StringVar(value="")
        self.entry_cand = tk.StringVar(value="")
        self.entry_cand_conf = tk.StringVar(value="")
        self.entry_notes = tk.StringVar(value="")
        self.save_status = tk.StringVar(value="")
        self.locked = tk.BooleanVar(value=False)
        self.form_widgets = []    # everything the lock disables
        super().__init__(app, parent)

    # -- layout ------------------------------------------------------------------------

    def build_cursor(self, parent):
        box = ttk.LabelFrame(parent, text="Cursor", padding=BOX_PAD)
        box.pack(fill=tk.X, pady=BOX_GAP)

        # readonly rather than disabled: the text can still be selected and copied, but not
        # typed over. The cursor is moved by clicking and scrolling the views.
        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Label(row, text="(x, y, z) = ").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.coord_var, state="readonly", width=15,
                  font=FIXED_FONT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(box, text="Position of the crosshair", foreground="#555",
                  font=HINT_FONT).pack(anchor=tk.W, pady=(1, 3))

        row_actual = ttk.Frame(box)
        row_actual.pack(fill=tk.X)
        ttk.Label(row_actual, text="Actual (x, y, z) = ").pack(side=tk.LEFT)
        ttk.Entry(row_actual, textvariable=self.actual_var, state="readonly", width=15,
                  font=FIXED_FONT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(box, text="Actual position in OG file before the flips",
                  foreground="#555", font=HINT_FONT).pack(anchor=tk.W, pady=(1, 3))

        row_rgb = ttk.Frame(box)
        row_rgb.pack(fill=tk.X)
        ttk.Label(row_rgb, text="(R, G, B) = ").pack(side=tk.LEFT)
        ttk.Entry(row_rgb, textvariable=self.rgb_var, state="readonly", width=18,
                  font=FIXED_FONT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(box, text="Channel values in the loaded file", foreground="#555",
                  font=HINT_FONT).pack(anchor=tk.W, pady=(1, 3))

        row2 = ttk.Frame(box)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Label under cursor:  label ").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.label_var, state="readonly", width=8,
                  font=FIXED_FONT).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row3 = ttk.Frame(box)
        row3.pack(fill=tk.X, pady=(ROW_GAP, 0))
        ttk.Label(row3, text="Find label/neuron ").pack(side=tk.LEFT)
        # Editable, so a name or an id can be typed as well as picked off the list.
        self.find_box = ttk.Combobox(row3, textvariable=self.find_var, width=10)
        self.find_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.find_box.bind("<<ComboboxSelected>>", lambda e: self.on_find())
        self.find_box.bind("<Return>", lambda e: self.on_find())

    def build_right_extras(self, parent):
        """The labelling form, under the contrast curve."""
        box = ttk.LabelFrame(parent, text="Label Neurons", padding=BOX_PAD)
        box.pack(side=tk.TOP, fill=tk.X, pady=(BOX_GAP * 2, 0))

        def line():
            row = ttk.Frame(box)
            row.pack(fill=tk.X, pady=(ROW_GAP, 0))
            return row

        def hint(text):
            ttk.Label(box, text=text, foreground="#555",
                      font=HINT_FONT).pack(anchor=tk.W)

        def field(row, var, width, **kwargs):
            entry = ttk.Entry(row, textvariable=var, width=width, **kwargs)
            self.form_widgets.append(entry)
            return entry

        top = line()
        ttk.Checkbutton(top, text="Lock", variable=self.locked,
                        command=self.apply_lock).pack(side=tk.LEFT)
        ttk.Label(top, text="  no edits while locked; labels still show",
                  foreground="#555", font=HINT_FONT).pack(side=tk.LEFT)
        ttk.Separator(box).pack(fill=tk.X, pady=(2, 0))

        self.build_label_display(box)
        ttk.Separator(box).pack(fill=tk.X, pady=(2, 0))

        # Label id and position track the crosshair, but stay editable: typing either one
        # and pressing Return moves the crosshair instead of letting the two disagree.
        row = line()
        ttk.Label(row, text="Label ").pack(side=tk.LEFT)
        id_entry = field(row, self.entry_id, 5)
        id_entry.pack(side=tk.LEFT)
        id_entry.bind("<Return>", lambda e: self.goto_typed_roi())
        ttk.Label(row, text="  Position ").pack(side=tk.LEFT)
        pos_entry = field(row, self.entry_pos, 12)
        pos_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        pos_entry.bind("<Return>", lambda e: self.goto_typed_position())

        row = line()
        ttk.Label(row, text="Neuron name ").pack(side=tk.LEFT)
        field(row, self.entry_conf, 4).pack(side=tk.RIGHT)
        ttk.Label(row, text=" Confidence ").pack(side=tk.RIGHT)
        field(row, self.entry_neuron, 8).pack(side=tk.LEFT, fill=tk.X, expand=True)
        hint(f"Confidence on a scale of {CONFIDENCE_RANGE[0]}-{CONFIDENCE_RANGE[1]}, "
             f"{CONFIDENCE_SPLIT_NOTE}")

        row = line()
        ttk.Label(row, text="Second Candidate ").pack(side=tk.LEFT)
        field(row, self.entry_cand_conf, 4).pack(side=tk.RIGHT)
        ttk.Label(row, text=" Confidence ").pack(side=tk.RIGHT)
        field(row, self.entry_cand, 8).pack(side=tk.LEFT, fill=tk.X, expand=True)
        hint("Second candidate can be left empty")

        row = line()
        ttk.Label(row, text="Notes ").pack(side=tk.LEFT)
        # the cap is enforced as you type, not only when you save
        allow = (self.app.root.register(lambda P: len(P) <= NOTES_MAX_CHARS), "%P")
        field(row, self.entry_notes, 10, validate="key", validatecommand=allow).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        hint(f"Optional, max {NOTES_MAX_CHARS} characters")

        save = ttk.Button(box, text="Save", command=self.on_save_label)
        save.pack(fill=tk.X, pady=(4, 0))
        self.form_widgets.append(save)
        hint('Click to save, hitting "return" on keyboard does not save')
        self.build_labels_path_row(box)

        ttk.Label(box, textvariable=self.save_status, justify=tk.LEFT,
                  foreground="#333", font=HINT_FONT,
                  wraplength=RIGHT_PANEL_WIDTH - 30).pack(anchor=tk.W, pady=(3, 0))

    def apply_lock(self):
        """Dim and disable the form. The label overlay is display, so it stays live."""
        state = ["disabled"] if self.locked.get() else ["!disabled"]
        for widget in self.form_widgets:
            widget.state(state)

    # -- the label table ---------------------------------------------------------------

    def on_volume_loaded(self):
        super().on_volume_loaded()
        self.shown_roi = None
        self.clear_form()
        self.refresh_find_choices()
        self.save_status.set(f"{os.path.basename(self.labels_path)}: "
                             f"{len(self.saved)} labelled ROI(s)")

    def roi_under_cursor(self):
        rois = self.app.companion_data.get("neuron_rois")
        if rois is None:
            return None
        x, y, z = self.cursor
        return int(rois[x, y, z])

    def clear_form(self):
        for var in (self.entry_neuron, self.entry_conf, self.entry_cand,
                    self.entry_cand_conf, self.entry_notes):
            var.set("")

    def show_saved(self, roi_id):
        """Put an ROI's stored row back in the form, or clear it if there is none."""
        row = self.saved.get(roi_id)
        if row is None:
            self.clear_form()
            self.save_status.set(
                "no ROI under the crosshair" if not roi_id
                else f"ROI {roi_id}: not yet labelled")
            return
        self.entry_neuron.set(row["Neuron Class"])
        self.entry_conf.set(row["Confidence"])
        self.entry_cand.set(row["Second Candidate"])
        self.entry_cand_conf.set(row["Second Candidate Confidence"])
        self.entry_notes.set(row["Notes"])
        if (row["Neuron Class"] or "").strip():
            self.save_status.set(f"ROI {roi_id}: saved as {row['Neuron Class']} "
                                 f"(confidence {row['Confidence']}) -- saving again edits "
                                 f"that row")
        else:
            self.save_status.set(f"ROI {roi_id}: a row is kept for it but it has no name "
                                 f"yet -- saving again edits that row")

    def refresh_find_choices(self):
        """List every ROI as "<id> <name>", with the name blank until one is given."""
        rois = self.app.companion_data.get("neuron_rois")
        if self.find_box is None or rois is None:
            return
        ids = np.unique(rois)
        self.find_box.configure(values=[
            f"{int(i)} {self.saved.get(int(i), {}).get('Neuron Class', '')}".rstrip() + " "
            for i in ids[ids > 0]])

    def on_find(self):
        """Jump the crosshair to the chosen ROI, in both views.

        Takes the leading id from a picked entry, and otherwise reads what was typed as a
        neuron name. Landing on the ROI sets all three coordinates, so the lateral view
        moves to its z slice and the coronal to its y.
        """
        text = self.find_var.get().strip()
        if not text:
            return
        head = text.split()[0]
        if head.lstrip("-").isdigit():
            self.entry_id.set(head)
            self.goto_typed_roi()
            return
        wanted = text.lower()
        for roi_id, row in self.saved.items():
            if (row["Neuron Class"] or "").strip().lower() == wanted:
                self.entry_id.set(str(roi_id))
                self.goto_typed_roi()
                return
        self.save_status.set(f"no ROI or neuron called {text!r}")

    def goto_typed_roi(self):
        """Jump the crosshair to a typed ROI id, landing on a voxel inside it."""
        rois = self.app.companion_data.get("neuron_rois")
        if rois is None:
            return
        try:
            roi_id = int(self.entry_id.get().strip())
        except ValueError:
            self.save_status.set("ROI id must be a whole number")
            return
        idx = np.argwhere(rois == roi_id)
        if not len(idx):
            self.save_status.set(f"no ROI {roi_id} in this volume")
            return
        # the centroid of a curved blob can fall outside it, so take the voxel nearest to it
        nearest = idx[np.argmin(((idx - idx.mean(axis=0)) ** 2).sum(axis=1))]
        self.cursor = [int(v) for v in nearest]
        self.refresh()

    def goto_typed_position(self):
        """Move the crosshair to a typed (x, y, z), read as source-file coordinates."""
        text = self.entry_pos.get().strip().strip("()")
        try:
            parts = [int(round(float(p))) for p in text.replace(",", " ").split()]
            if len(parts) != 3:
                raise ValueError
        except ValueError:
            self.save_status.set("position must be three numbers, e.g. 142, 60, 32")
            return
        parts = [int(np.clip(v, 0, self.app.raw.shape[i] - 1)) for i, v in enumerate(parts)]
        self.cursor = self.app.to_source_coords(parts)      # back into display indices
        self.refresh()

    def on_save_label(self):
        """Validate the form and write it into the label table."""
        if self.app.data is None or self.labels_path is None or self.locked.get():
            return
        roi_id = self.roi_under_cursor()
        try:
            roi_id = int(self.entry_id.get().strip())
        except ValueError:
            pass
        if not roi_id:
            messagebox.showwarning(
                "No ROI", "The crosshair is on background (ROI 0).\n\n"
                "Rows are keyed by ROI id, so put the crosshair inside an ROI first.")
            return

        # An empty name is allowed: it records the ROI without identifying it, which is
        # how a name gets cleared again. Confidence is only required alongside a name.
        neuron = self.entry_neuron.get().strip()
        conf = self.validated_confidence(self.entry_conf.get(), required=bool(neuron))
        if conf is None:
            return
        cand = self.entry_cand.get().strip()
        cand_conf = ""
        if self.entry_cand_conf.get().strip():
            cand_conf = self.validated_confidence(self.entry_cand_conf.get(),
                                                  required=False)
            if cand_conf is None:
                return

        notes = self.entry_notes.get().strip()[:NOTES_MAX_CHARS]
        existed = roi_id in self.saved
        self.saved[roi_id] = {
            "Neuron Class": neuron,
            "Coordinates": self.entry_pos.get().strip().strip("()").replace(" ", ""),
            "ROI ID": str(roi_id),
            "Confidence": str(conf),
            "Second Candidate": cand,
            "Second Candidate Confidence": str(cand_conf),
            "Notes": notes,
        }
        try:
            write_labels(self.labels_path, self.saved)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.shown_roi = roi_id
        self.refresh_find_choices()
        named = sum(1 for r in self.saved.values() if (r["Neuron Class"] or "").strip())
        what = f"as {neuron}" if neuron else "with no name (kept for later)"
        self.save_status.set(
            f"ROI {roi_id}: {'updated' if existed else 'added'} {what} -- "
            f"{named} named of {len(self.saved)} rows, in "
            f"{os.path.basename(self.labels_path)}")
        self.refresh()      # so a new name shows at once when the overlay is on

    def validated_confidence(self, text, required):
        """Confidence must be a whole number in range, or the split-ROI flag.

        Returns None once it has complained, so callers can just bail on None.
        """
        lo, hi = CONFIDENCE_RANGE
        text = text.strip()
        if not text:
            if required:
                messagebox.showwarning(
                    "Missing confidence",
                    f"Confidence is required, on a scale of {lo}-{hi}, "
                    f"{CONFIDENCE_SPLIT_NOTE}.")
                return None
            return ""
        try:
            value = int(text)
        except ValueError:
            value = None
        if value is None or not (lo <= value <= hi or value == CONFIDENCE_SPLIT_ROI):
            messagebox.showwarning(
                "Bad confidence",
                f"Confidence must be a whole number from {lo} to {hi}, "
                f"{CONFIDENCE_SPLIT_NOTE}.")
            return None
        return value

    # -- readouts ----------------------------------------------------------------------

    def after_draw(self, pcts):
        x, y, z = self.cursor
        self.coord_var.set(f"({x}, {y}, {z})")
        ax, ay, az = self.app.to_source_coords(self.cursor)
        self.actual_var.set(f"({ax}, {ay}, {az})")
        # The file's own values, not the displayed ones: the brightness percentages and the
        # contrast curve change what a neuron looks like, never what it is.
        voxel = self.app.data[x, y, z]
        self.rgb_var.set("(" + ", ".join(f"{float(voxel[idx]):.1f}"
                                         for _, _, idx in CHANNELS) + ")")
        roi_id = self.roi_under_cursor()
        self.label_var.set("-" if roi_id is None else str(roi_id))

        # The form's id and position mirror the crosshair. Position carries the source
        # file's coordinates rather than the displayed ones, because that is what the
        # Coordinates column records -- the box shows exactly what a save would write.
        self.entry_id.set("" if roi_id is None else str(roi_id))
        self.entry_pos.set(f"{ax}, {ay}, {az}")

        # Recall or clear the rest only when the ROI actually changes, so nudging the
        # crosshair inside one ROI never wipes what is being typed.
        if roi_id is not None and roi_id != self.shown_roi:
            self.shown_roi = roi_id
            self.show_saved(roi_id)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

class NeuroPALGUI:
    """The window: loads the volumes and owns the display orientation.

    Processing dials the channel balance and writes; Labeling steps through slices to
    locate neurons and records their identities. Both read the same loaded volume.
    """

    def __init__(self, root, neuropal_path=None, all_red_path=None, neuron_rois_path=None,
                 save_dir=None, labels_path=None):
        self.root = root
        self.root.title("NeuroPAL")
        use_compact_style(root)

        self.src_path = None      # path of the loaded file
        self.raw = None           # pristine volume, in the file's own coordinates
        self.data = None          # `raw` with the display flips applied
        self.header = None
        self.ref = 1.0            # display white point, shared by the channels and tabs

        # Companion volumes, read for the overlays. Kept unflipped in `*_raw` and flipped
        # alongside `data`, so the overlays stay voxel-aligned on screen.
        self.companion_raw = {name: None for name in COMPANION_VOLUMES}
        self.companion_data = {name: None for name in COMPANION_VOLUMES}
        self.companion_paths = {}
        self.all_red_ref = 1.0    # white point for the grayscale overlay
        self.roi_max = 1          # largest ROI id, so colours are stable across slices
        self.roi_centroids = None # lazily built; invalidated whenever the flips change

        # Where output goes, and which label table to use. Either may be set by the
        # launcher; otherwise they follow the NeuroPAL file. Browsing a table in either tab
        # replaces `labels_override`, so the choice sticks across reloads.
        self.save_dir = save_dir
        self.labels_override = labels_path

        self.path_var = tk.StringVar(value=neuropal_path or "")
        self.companion_vars = {"all_red": tk.StringVar(value=all_red_path or ""),
                               "neuron_rois": tk.StringVar(value=neuron_rois_path or "")}
        # Display orientation, shared by both tabs and driven by an anatomical code. The
        # per-axis flips stay the mechanism underneath -- everything downstream, including
        # the save, already speaks in those terms. No flips on startup, so a freshly loaded
        # file is shown exactly as it sits on disk.
        self.flips = {key: tk.BooleanVar(value=False) for _, key, _ in FLIPS}
        self.source_code = "LPS"            # what the file declares
        self.orientation_var = tk.StringVar(value="LPS")
        self.source_note_var = tk.StringVar(value="")
        self.orientation_boxes = []
        self.status_var = tk.StringVar(value="No file loaded.")
        self.output_var = tk.StringVar(value="")

        self._build()
        if neuropal_path:
            self.load()

    def _build(self):
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=4).pack(side=tk.BOTTOM, fill=tk.X)

        # Notebook at the very top, so the tabs sit in the upper left of the window.
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.processing_tab = ProcessingTab(self, self.notebook)
        self.label_tab = LabelingTab(self, self.notebook)
        self.notebook.add(self.processing_tab.frame, text="Processing")
        self.notebook.add(self.label_tab.frame, text="Labeling")

        # A tab only redraws while it is visible, so catch up when one is selected.
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.on_tab_changed())
        self.describe_output()
        self.refresh_orientation()

    def on_tab_changed(self):
        """Redraw the tab being shown, and give it the keyboard if it wants it."""
        current = self.current_tab()
        for tab in self.tabs:
            if tab is not current:
                tab.release_keys()
        current.claim_keys()
        current.reload_labels()     # names saved in the other tab show up here
        current.refresh_if_stale()

    @property
    def tabs(self):
        return [self.processing_tab, self.label_tab]

    def current_tab(self):
        selected = self.notebook.select()
        for tab in self.tabs:
            if str(tab.frame) == selected:
                return tab
        return self.processing_tab

    # -- paths -------------------------------------------------------------------------

    def output_dir(self):
        """Where the corrected volumes and the label table go."""
        if self.save_dir:
            return os.path.abspath(self.save_dir)
        return default_output_dir(self.src_path)

    def labels_file(self):
        if self.labels_override:
            return os.path.abspath(self.labels_override)
        return labels_path_for(self.src_path, self.output_dir())

    def roi_centroid(self, roi_id):
        """The 3D centroid of an ROI in the displayed volume, or None.

        Built for every ROI at once the first time it is asked for -- one pass beats
        scanning the volume per label, which is what a projection full of names would
        otherwise cost on every redraw.
        """
        rois = self.companion_data.get("neuron_rois")
        if rois is None:
            return None
        if self.roi_centroids is None:
            ids = np.unique(rois)
            ids = ids[ids > 0]
            ones = np.ones(rois.shape, dtype=np.float32)
            centres = ndimage.center_of_mass(ones, labels=rois, index=ids)
            self.roi_centroids = {int(i): c for i, c in zip(ids, centres)}
        return self.roi_centroids.get(int(roi_id))

    def current_orientation_code(self):
        """The code the current flips amount to."""
        flips = tuple(bool(self.flips[key].get()) for _, key, _ in FLIPS)
        for code, combination, _ in orientation_options(self.source_code):
            if combination == flips:
                return code
        return self.source_code

    def orientation_choices(self):
        """The codes to offer: rotations of the source only, never a reflection."""
        return [code for code, _, mirrored in orientation_options(self.source_code)
                if not mirrored]

    def refresh_orientation(self):
        """Rebuild the choices for the loaded file and show the code in force."""
        choices = self.orientation_choices()
        for combo in self.orientation_boxes:
            combo.configure(values=choices)
        current = self.current_orientation_code()
        if current in choices:
            self.orientation_var.set(current)
        elif choices:
            self.orientation_var.set(choices[0])
        self.source_note_var.set(f"file is {self.source_code}")

    def on_orientation_selected(self):
        """A code was picked: turn it into flips and redraw."""
        code = self.orientation_var.get().split()[0]
        for key, value in flips_for_orientation(self.source_code, code).items():
            self.flips[key].set(value)
        self.on_flip()

    def to_source_coords(self, coords):
        """Map a voxel between the displayed volume and the source file's own indices.

        A flip is its own inverse, so the same call converts either way: display -> source
        for saving, source -> display for putting the crosshair somewhere.
        """
        out = list(coords)
        for axis, key, _ in FLIPS:
            if self.flips[key].get():
                out[axis] = self.raw.shape[axis] - 1 - out[axis]
        return out

    def flip_state(self):
        return {key: bool(var.get()) for key, var in self.flips.items()}

    def describe_output(self):
        if self.src_path is None:
            self.output_var.set("writes NeuroPAL_corrected.nrrd, the companions and "
                                "neuropal_info.txt")
            return
        self.output_var.set(f"writes to {self.output_dir()}\n"
                            "NeuroPAL_corrected.nrrd, all_red_corrected.nrrd, "
                            "neuron_rois_corrected.nrrd, neuropal_info.txt\n"
                            "saved in the orientation on screen; neuropal_info.txt "
                            "records the code")

    # -- loading -----------------------------------------------------------------------

    def show_path_tail(self, entry, var):
        """Keep an entry scrolled to the end of its path.

        Re-applied on <Configure> as well as on a change, because at build time the entry
        has no width yet and scrolling it then does nothing. Skipped while it has focus, so
        it never yanks the view away from someone editing the path by hand.
        """
        def to_end(*_):
            if entry.focus_get() is not entry:
                entry.after_idle(lambda: entry.xview_moveto(1.0))
        var.trace_add("write", to_end)
        entry.bind("<Configure>", to_end)
        to_end()

    def browse(self, var, label):
        path = filedialog.askopenfilename(
            title=f"Select {label}",
            filetypes=[("NRRD volumes", "*.nrrd"), ("All files", "*.*")])
        if path:
            var.set(path)

    def load(self):
        path = self.path_var.get().strip()
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showerror("Load failed", f"No such file:\n{path}")
            return
        try:
            self.raw, self.header = load_neuropal(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.src_path = path
        self.source_code = source_orientation(self.header)
        self.refresh_orientation()
        self.ref = display_reference(self.raw)
        self.resolve_companions()
        self.load_companion_volumes()
        self.apply_flips_to_volumes()
        self.describe_output()
        for tab in self.tabs:
            tab.on_volume_loaded()
        self.status_var.set(
            f"Loaded {os.path.basename(path)}  {self.raw.shape} {self.raw.dtype}  "
            f"(display white point = p{DISPLAY_PERCENTILE:g} of raw MIP = {self.ref:.0f})")

    def resolve_companions(self):
        """Explicit companion paths if the boxes hold any, else beside the NeuroPAL file."""
        given = {name: self.companion_vars[name].get().strip() or None
                 for name in COMPANION_VOLUMES}
        self.companion_paths = resolve_companion_paths(
            self.src_path, all_red_path=given["all_red"],
            neuron_rois_path=given["neuron_rois"])
        for name in COMPANION_VOLUMES:
            self.companion_vars[name].set(self.companion_paths[name])

    def load_companion_volumes(self):
        """Read all_red / neuron_rois for the overlays, if they are where we expect them.

        A missing or mismatched companion is not an error -- the overlay is simply
        unavailable, and both tabs grey its checkbox out.
        """
        for name in COMPANION_VOLUMES:
            self.companion_raw[name] = None
            path = self.companion_paths.get(name)
            if not path or not os.path.exists(path):
                continue
            try:
                vol, _ = nrrd.read(path)
            except Exception:
                continue
            if vol.shape[:3] != self.raw.shape[:3]:
                continue
            self.companion_raw[name] = vol

        red = self.companion_raw["all_red"]
        if red is not None:
            ref = float(np.percentile(red, DISPLAY_PERCENTILE))
            self.all_red_ref = ref if ref > 0 else float(red.max()) or 1.0
        rois = self.companion_raw["neuron_rois"]
        self.roi_max = int(rois.max()) if rois is not None and rois.max() > 0 else 1

    def apply_flips_to_volumes(self):
        """Re-derive the on-screen volumes from the pristine copies, so flips never stack."""
        fx, fy, fz = (bool(self.flips[k].get()) for _, k, _ in FLIPS)
        self.data = apply_flips(self.raw, fx, fy, fz)
        for name in COMPANION_VOLUMES:
            vol = self.companion_raw[name]
            self.companion_data[name] = (None if vol is None
                                         else apply_flips(vol, fx, fy, fz))
        self.roi_centroids = None   # they move with the volume

    def on_flip(self):
        """The display orientation changed: re-derive and redraw both tabs."""
        self.refresh_orientation()
        if self.raw is None:
            return
        self.apply_flips_to_volumes()
        for tab in self.tabs:
            tab.invalidate()


def main(neuropal_path=None, all_red_path=None, neuron_rois_path=None,
         save_dir=None, labels_path=None):
    """Build the window and run the Tk event loop.

    All five paths are optional. `all_red_path` / `neuron_rois_path` override where the
    companions are read from; leave them None to look beside the NeuroPAL file. `save_dir`
    overrides where the corrected volumes and the label table are written, and
    `labels_path` points at an existing label table instead of starting a new one.
    """
    root = tk.Tk()
    NeuroPALGUI(root, neuropal_path=neuropal_path, all_red_path=all_red_path,
                neuron_rois_path=neuron_rois_path, save_dir=save_dir,
                labels_path=labels_path)
    # Fill the display rather than guessing a size: the panels are a fixed width, so every
    # extra pixel of window width goes straight into the lateral and coronal images. On
    # Windows, maximise properly instead -- winfo_screenheight() there counts the taskbar,
    # so sizing to it would push the bottom of the window off screen.
    root.update_idletasks()
    if sys.platform.startswith("win"):
        root.state("zoomed")
    else:
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    root.mainloop()
