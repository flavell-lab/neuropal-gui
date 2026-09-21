# NeuroPAL GUI

An interactive viewer for NeuroPAL volumes with two jobs: balance the colour channels of a
multichannel `.nrrd`, and identify neurons ROI by ROI into a label table. Both tabs work on
the same loaded volume and show a lateral and a coronal panel with a linked crosshair.

Built on Tk + matplotlib; volumes are read and written with `pynrrd`.

## Get the code

Clone the repository and move into the folder that creates. 

```bash
git clone https://github.com/hongyin-liu/neuropal-gui.git
cd neuropal-gui
```

That prompts for your GitHub username and a
[personal access token](https://github.com/settings/tokens) as the password. If you already have an SSH key registered with GitHub, this clones without a prompt instead:

```bash
git clone git@github.com:hongyin-liu/neuropal-gui.git
cd neuropal-gui
```

Every command below is run from inside that `neuropal-gui` folder.

## Install and run

With [uv](https://docs.astral.sh/uv/) (recommended — the Python it downloads includes Tk).
If you do not have uv, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` on
macOS or Linux, or see its [install page](https://docs.astral.sh/uv/getting-started/installation/)
for Windows. Then:

```bash
uv sync
uv run run_neuropal_gui_example.py
```

Or with an existing Python 3.10+ that has `tkinter`:

```bash
pip install matplotlib numpy pynrrd scipy
python3 run_neuropal_gui_example.py
```

This opens the bundled example dataset in `2023-01-06-08/`. To use your own data, edit the
paths at the top of `run_neuropal_gui_example.py`, or leave them and use the **Browse**
boxes in the Processing tab.

## Workflow

Two passes over a dataset:

1. **Processing** — set the display orientation and the per-channel brightness, then save.
   The orientation and the brightness are baked into the written volumes.
2. **Labeling** — reopen the corrected volumes and name neurons. Positions are recorded in the loaded file's own coordinates.

## Processing tab

![Processing tab](gui_screenshots/screenshot_mip.png)

Top row: paths to `NeuroPAL.nrrd` and its companions `all_red.nrrd` and `neuron_rois.nrrd`,
then **Load**. Both panels are maximum intensity projections — lateral over z, coronal
over y.

- **Display orientation** — pick the anatomical code (LPS, RAS, …) the volume should be
  shown in. Only rotations of the file's own orientation are offered; reflections would
  swap left and right. Aim for the ventral nerve cord at the bottom of the lateral view,
  which puts left-side neurons at the bottom of the coronal view.
- **Channel brightness** — one slider and box per channel, as a percentage of a division
  factor (100% = untouched, 50% = divided by 2, 200% = multiplied by 2). The checkboxes
  turn a channel, or the whole NeuroPAL image, off in the display.
- **Cursor** — voxel coordinates under the crosshair with the raw and corrected channel
  values. Click or drag either panel to move it.
- **Overlays** — `all_red` as a grey overlay, `neuron_rois` as a colour-coded segmentation,
  each with an opacity box; `a` and `s` toggle them.
- **Curve-Based Contrast Adjustment** — an ITK-SNAP-style transfer curve over the intensity
  histogram. Drag the control points; display only, never written to file.
- **Output** — **Save corrected NRRDs + info** writes, next to the source by default:
  `NeuroPAL_corrected.nrrd` (reoriented and rescaled), `all_red_corrected.nrrd` and
  `neuron_rois_corrected.nrrd` (reoriented only), and `neuropal_info.txt` recording the
  saved orientation and the factors used.
- **Labeled Neurons (Optional)** — draw the names from an existing label table over the
  images, filtered by a confidence threshold (`n` toggles them).

## Labeling tab

![Labeling tab with the segmentation overlay](gui_screenshots/screenshot_roi.png)

Same volume, one z slice at a time instead of a projection, and nothing here can write a
volume. Drag the scrollbars beside each panel to step through z (lateral) and y (coronal).
The left panel repeats the orientation dropdown, the channel brightness and the overlays,
plus a fuller cursor readout:

- **(x, y, z)** — the crosshair position, in the orientation on screen.
- **Actual (x, y, z)** — the same point in the loaded file's own indices, which is what
  gets saved.
- **(R, G, B)** — the channel values at that voxel.
- **Label under cursor** — the `neuron_rois` ROI id under the crosshair.

![Labeling tab with saved names drawn on the image](gui_screenshots/screenshot_id.png)

**Label Neurons**, on the right, records one neuron per ROI:

- **Label** and **Position** fill themselves in from the crosshair.
- **Neuron name**, **Confidence** (1–5, or −1 for a split ROI merged at segmentation),
  an optional **Second Candidate** with its own confidence, and **Notes** (max 300 chars).
- **Save** writes the row — it has to be clicked; pressing Return does not save. Saving an
  ROI again edits its row in place rather than appending a second one.
- **Lock** blocks edits while leaving the names visible; **Show label on image** and the
  **Confidence threshold** control which names are drawn.
- **Path** is the label table, created on the first save as
  `<dataset folder>_labels.csv` in the output folder.

### Finding a neuron

![Find label/neuron dropdown](gui_screenshots/screenshot_search.png)

**Find label/neuron** lists every saved label as `<ROI id> <name>`. Pick one off the list,
or type an ROI id or a neuron name and press Return: the crosshair jumps to that ROI's
centroid in both panels, changing the z and y slices as needed.

## Label table

One row per ROI, written as UTF-8 CSV with these columns:

`Neuron Class`, `Coordinates`, `ROI ID`, `Confidence`, `Second Candidate`,
`Second Candidate Confidence`, `Notes`

Named rows come first, sorted by ROI id, then any rows left unidentified. Recall keys are on `ROI ID` (used for displaying neuron names on roi), so `Coordinates` is written for user's reference only.

## Notes

- Runs on macOS, Linux and Windows.
- Over SSH the GUI needs a display: `ssh -X` (X11 forwarding) works but is slow with large
  volumes, so a remote desktop session (VNC/X2Go) or copying the data locally is smoother.
- Volumes are `(X, Y, Z, C)` for NeuroPAL and `(X, Y, Z)` for the companion files.

## License

MIT — see [LICENSE](LICENSE).
