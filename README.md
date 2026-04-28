# NeuroFlow — EEG Preprocessing & Analysis Dashboard

A production-grade EEG analysis tool built with Streamlit and MNE-Python.

## Features

- **Multi-file upload** — drop multiple EDF files and switch between them instantly
- **Full preprocessing pipeline** — bandpass filter → notch filter → ICA artifact removal → average re-reference
- **Configurable parameters** — adjust all filter frequencies and ICA threshold from the sidebar
- **Cognitive state analysis** — frequency band power distribution with dominant-state inference
- **Topographic maps** — interactive scalp potential maps with a time slider
- **Export everything**
  - ✅ Clean EDF file
  - ✅ Excel workbook (3 sheets: time-series, channel stats, band powers)
  - ✅ CSV (flat µV time-series)
  - ✅ PNG plots (PSD, time-series, topomap)
  - ✅ Batch ZIP export for multiple files

---

## Local Setup

```bash
# 1. Clone / copy files into a folder
# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Free Hosting Options

### Option A — Hugging Face Spaces ⭐ Recommended
Best free RAM headroom for heavy scientific libraries like MNE.

1. Create an account at https://huggingface.co
2. New Space → SDK: **Streamlit** → Public
3. Upload `app.py`, `eeg_pipeline.py`, `requirements.txt`
4. Your app is live at `https://huggingface.co/spaces/<username>/<space-name>`

### Option B — Streamlit Community Cloud
1. Push files to a **public GitHub repo**
2. Go to https://share.streamlit.io → New app → point to `app.py`
3. Live at `https://<yourapp>.streamlit.app`
> ⚠️ Free tier has ~1 GB RAM — MNE can be heavy with long recordings.

### Option C — Render.com
More control; good for larger files.
1. Add a `render.yaml` or use the web service wizard
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## File Structure

```
├── app.py              ← Streamlit UI (main entry point)
├── eeg_pipeline.py     ← All EEG processing logic (MNE)
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## EDF Channel Support

The app expects EMOTIV EPOC X channels by default:
`AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`

If your EDF uses different channel names, the app automatically falls back to the first 14 channels found.
