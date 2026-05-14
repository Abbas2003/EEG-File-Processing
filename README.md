# NeuroFlow — Unsupervised EEG Cognitive Analysis

NeuroFlow is a production-grade EEG analysis suite built with Streamlit, MNE-Python, and Scikit-Learn. It allows you to transform raw EEG data into actionable cognitive insights using unsupervised machine learning.

## Key Features

- **Full Preprocessing Pipeline** — Bandpass filter → Notch filter → ICA artifact removal → Average re-reference.
- **Signal Quality Guard** — Real-time detection of flatlines, clipping, and SNR improvement reporting.
- **Automated Epoching** — High-resolution segmentation into 4-second windows with 50% overlap.
- **Deep Feature Extraction** — Extracts ~370 features per epoch (Statistical, Hjorth, Spectral Entropy, Band Ratios, Frontal Asymmetry).
- **Unsupervised Discovery** — Automatically clusters unlabeled EEG data to discover patterns matching Focused, Distracted, and Anxious states.
- **Cognitive Timeline** — Visualize state transitions over time in a Gantt-style chart.
- **Advanced Export** — Export clean EDFs, Feature Matrices (CSV), and Clustered Results with interpreted labels.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_app.py
```

## Scientific Foundation

- **Frontal Alpha Asymmetry (FAA)**: Used as a proxy for emotional valence and approach/withdrawal motivation.
- **Engagement Index**: Calculated as `Beta / (Alpha + Theta)`, a validated measure of cognitive focus.
- **Hjorth Parameters**: Quantify signal activity, mobility, and complexity.
- **Fractal Dimension**: Measures signal self-similarity using the Higuchi method.

## File Structure

```
├── streamlit_app.py   ← Main UI & Dashboard
├── eeg_pipeline.py    ← Core MNE processing & quality metrics
├── eeg_epochs.py      ← Fixed-length segmentation logic
├── eeg_features.py    ← 370+ feature extraction algorithms
├── eeg_clustering.py  ← ML clustering & interpretation logic
└── requirements.txt   ← Python dependencies
```
