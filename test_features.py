import mne
import numpy as np
import pandas as pd
from eeg_features import extract_all_features

# Create dummy data
info = mne.create_info(ch_names=['AF3', 'AF4', 'F3', 'F4'], sfreq=128, ch_types='eeg')
data = np.random.randn(1, 4, 128 * 4) * 1e-5 # 1 epoch, 4 channels, 4 seconds
epochs = mne.EpochsArray(data, info)

try:
    print("Starting feature extraction test...")
    df = extract_all_features(epochs)
    print(f"Success! Extracted {df.shape[1]} features.")
    print("Columns:", df.columns[:10].tolist())
except Exception as e:
    print(f"Error during feature extraction: {e}")
    import traceback
    traceback.print_exc()
