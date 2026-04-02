import os
import numpy as np
import mne
import warnings

# We ignore warnings to keep the output clean, especially from MNE which can be verbose
warnings.filterwarnings("ignore")


def load_epochs_from_edf(filepath, tmin=-0.5, tmax=4.0, plot=False):
    """
    Load an EDF file, apply band-pass filtering, extract epochs for T1 and T2 events,
    and return the data as numpy arrays.
    """
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    if plot:
        raw.plot(duration=10, n_channels=5, title=f"Raw Data - {os.path.basename(filepath)}", block=False)

    raw.filter(l_freq=8.0, h_freq=30.0, fir_design="firwin", verbose=False)

    if plot:
        raw.plot(duration=10, n_channels=5, title=f"Filtered Data (8-30Hz) - {os.path.basename(filepath)}", block=True)

    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    target_events = {}
    if "T1" in event_dict:
        target_events["T1"] = event_dict["T1"]
    if "T2" in event_dict:
        target_events["T2"] = event_dict["T2"]

    if not target_events:
        return None, None

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(
        raw, events, event_id=target_events,
        tmin=tmin, tmax=tmax, picks=picks,
        baseline=(None, 0), preload=True, verbose=False
    )

    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    return X, y


def load_subject_epochs(subject_id, runs, base_path=None, plot=False):
    """
    Load and process data for a specific subject and specified runs.
    """
    if base_path is None:
        base_path = os.getenv("EEG_DATA_PATH", "data/files")

    X_list = []
    y_list = []

    subject_folder = f"S{subject_id:03d}"
    subject_path = os.path.join(base_path, subject_folder)

    print(f"\n--- Loading subject {subject_folder} | runs: {runs} ---")

    for run in runs:
        filepath = os.path.join(subject_path, f"{subject_folder}R{run:02d}.edf")

        if not os.path.exists(filepath):
            print(f"[-] Missing file: {filepath}")
            continue

        show_plot = plot and len(X_list) == 0
        X_run, y_run = load_epochs_from_edf(filepath, plot=show_plot)

        if X_run is not None and y_run is not None:
            X_list.append(X_run)
            y_list.append(y_run)
            print(f"[+] Run {run:02d} loaded: {X_run.shape[0]} epochs")

    if len(X_list) > 0:
        X_total = np.concatenate(X_list, axis=0)
        y_total = np.concatenate(y_list, axis=0)
        return X_total, y_total

    return None, None
