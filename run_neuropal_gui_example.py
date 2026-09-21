"""Launcher for the NeuroPAL GUI, set up to open the bundled example dataset.

Run it from a terminal:

    uv run run_neuropal_gui_example.py        # or: python3 run_neuropal_gui_example.py

A plain .py launcher, not an interactive session: Tk has to own the main thread and run
its own event loop. Started this way the GUI is stable, and closing the window returns
the shell.

To use your own data, point HOME at your dataset folder and edit the file names below.
Any path may be set to None to fall back to the default: the companion volumes are then
looked for beside the NeuroPAL file, output goes to a corrected_nrrds_and_labels/ folder
there, and the label table is created inside it.
"""

import os

from neuropal_gui import main

# The dataset folder. Everything below hangs off it, so switching datasets is one edit.
# Relative to this file, so the example runs from any working directory.
# Feel free to point this to your own dataset folder (can do the same for the nrrd files at your discretion).
HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2023-01-06-08")

# The example NeuroPAL volume has already been through the Processing tab, so it opens
# ready to label. Point NEUROPAL at a raw NeuroPAL.nrrd to balance the channels yourself.
NEUROPAL = f"{HOME}/NeuroPAL-corrected.nrrd"
ALL_RED = f"{HOME}/all_red.nrrd"
NEURON_ROIS = f"{HOME}/neuron_rois.nrrd"

# Where the corrected volumes, neuropal_info.txt and the label table are written. 
# Created if it does not exist yet.
SAVE_DIR = f"{HOME}/corrected_nrrds_and_labels"

# An existing label table to carry on from. 
# Leave as None to use <SAVE_DIR>/<dataset folder>_labels.csv, which is created on the first save.
# Note that saving rewrites the whole file. Recommend creating a copy of the original and working on that, so you can always go back to the original.

LABELS = f"{HOME}/2023-01-06-08 Neuron ID_labels_v1.csv" 
#LABELS = None  # Uncomment to start a new label table instead of continuing from the example.

if __name__ == "__main__":
    main(neuropal_path=NEUROPAL,
         all_red_path=ALL_RED,
         neuron_rois_path=NEURON_ROIS,
         save_dir=SAVE_DIR,
         labels_path=LABELS)
