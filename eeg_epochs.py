import mne
import numpy as np
import pandas as pd

def create_fixed_epochs(raw: mne.io.Raw, duration: float = 4.0, overlap: float = 2.0) -> mne.Epochs:
    """
    Segment continuous EEG into fixed-length epochs.
    - duration: length of each window in seconds
    - overlap: step size = duration - overlap
    """
    # 1. MNE Automatic Method
    raw_temp = raw.copy()
    raw_temp.set_annotations(None)
    events = mne.make_fixed_length_events(raw_temp, duration=duration, overlap=overlap)
    attempted_count = len(events)
    
    # 2. Manual Generator (Commented out)
    # sfreq = raw.info['sfreq']
    # total_duration = raw.times[-1]
    # step = duration - overlap
    # event_list = []
    # current_time = 0.0
    # while (current_time + duration) <= total_duration:
    #     sample_idx = int(current_time * sfreq) + raw.first_samp
    #     event_list.append([sample_idx, 0, 1])
    #     current_time += step
    # events = np.array(event_list)
    
    # Create epochs
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=duration, 
        baseline=None, preload=True, verbose=False,
        on_missing='ignore', reject_by_annotation=False
    )
    
    return epochs, attempted_count

def reject_bad_epochs(epochs: mne.Epochs, threshold_uv: float = 150.0) -> mne.Epochs:
    """
    Remove epochs where any channel exceeds the peak-to-peak threshold.
    threshold_uv: threshold in microvolts
    """
    # MNE threshold is in Volts
    reject_criteria = dict(eeg=threshold_uv * 1e-6)
    
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=reject_criteria, verbose=False)
    
    return epochs_clean

def get_epoch_summary(epochs: mne.Epochs, attempted: int = 0) -> dict:
    """Return stats about epoching and rejection."""
    kept = len(epochs)
    dropped = attempted - kept
    rejection_rate = (dropped / attempted * 100) if attempted > 0 else 0
    
    # Extract drop reasons from log
    # drop_log is a list of lists of strings
    reasons = []
    for log in epochs.drop_log:
        if log:
            reasons.extend(log)
    unique_reasons = list(set(reasons)) if reasons else ["None"]
    
    return {
        "total_epochs": attempted,
        "kept_epochs": kept,
        "dropped_epochs": dropped,
        "rejection_rate_pct": round(rejection_rate, 2),
        "drop_reasons": ", ".join(unique_reasons),
        "selection": epochs.selection.tolist() if hasattr(epochs, 'selection') else list(range(kept))
    }

def epochs_to_array(epochs: mne.Epochs) -> tuple[np.ndarray, list[float]]:
    """
    Returns (n_epochs, n_channels, n_samples) array 
    and the start time of each epoch.
    """
    data = epochs.get_data(copy=True)
    times = epochs.events[:, 0] / epochs.info['sfreq']
    return data, times.tolist()
