import mne
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
EEG_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

FREQ_BANDS = {
    'Delta':  (1,  4),
    'Theta':  (4,  8),
    'Alpha':  (8,  13),
    'Beta':   (13, 30),
    'Gamma':  (30, 45),
}

BAND_DESCRIPTIONS = {
    'Delta':  'Deep sleep & restorative processes',
    'Theta':  'Drowsiness, meditation & creativity',
    'Alpha':  'Relaxed alertness, eyes-closed rest',
    'Beta':   'Active thinking, focus & problem-solving',
    'Gamma':  'High-level information processing',
}

BAND_COLORS = {
    'Delta': '#6366f1',
    'Theta': '#8b5cf6',
    'Alpha': '#06b6d4',
    'Beta':  '#10b981',
    'Gamma': '#f59e0b',
}


# ──────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────
def load_raw_data(file_path: str) -> mne.io.Raw:
    """Load an EDF file, pick EEG channels, set montage."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    existing = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    if not existing:
        # fall back: take first 14 channels whatever they're called
        existing = raw.ch_names[:14]

    raw.pick_channels(existing)
    raw.set_channel_types({ch: 'eeg' for ch in existing})

    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
    except Exception:
        pass  # montage is cosmetic; carry on

    return raw


# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────
def apply_filtering(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_freq: float = 50.0,
) -> mne.io.Raw:
    """Bandpass + notch filter."""
    raw_f = raw.copy()
    raw_f.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    raw_f.notch_filter(freqs=notch_freq, fir_design='firwin', verbose=False)
    return raw_f


def run_ica_analysis(
    raw: mne.io.Raw,
    n_components: int | None = None,
    threshold: float = 2.5,
) -> mne.preprocessing.ICA:
    """Fit ICA and flag ocular / muscle artifact components."""
    n_comp = n_components or min(len(raw.ch_names), 15)
    ica = mne.preprocessing.ICA(
        n_components=n_comp, random_state=97, method='fastica', verbose=False
    )
    ica.fit(raw, verbose=False)

    frontal = [ch for ch in ['AF3', 'AF4', 'F7', 'F8'] if ch in raw.ch_names]
    if frontal:
        try:
            eog_idx, _ = ica.find_bads_eog(
                raw, ch_name=frontal, threshold=threshold, verbose=False
            )
            ica.exclude = eog_idx
        except Exception:
            pass

    return ica


def get_cleaned_data(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> mne.io.Raw:
    """Apply ICA + average re-reference."""
    raw_c = raw.copy()
    ica.apply(raw_c, verbose=False)
    raw_c.set_eeg_reference('average', projection=False, verbose=False)
    return raw_c


def full_pipeline(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_freq: float = 50.0,
    ica_threshold: float = 2.5,
):
    """One-shot preprocessing: filter → ICA → clean. Returns (raw_clean, ica)."""
    raw_f = apply_filtering(raw, l_freq, h_freq, notch_freq)
    ica   = run_ica_analysis(raw_f, threshold=ica_threshold)
    raw_c = get_cleaned_data(raw_f, ica)
    return raw_c, ica


# ──────────────────────────────────────────────
# VISUALISATION HELPERS
# ──────────────────────────────────────────────
def _dark_style():
    plt.rcParams.update({
        'figure.facecolor':  '#0d1117',
        'axes.facecolor':    '#161b22',
        'axes.edgecolor':    '#30363d',
        'axes.labelcolor':   '#c9d1d9',
        'text.color':        '#c9d1d9',
        'xtick.color':       '#8b949e',
        'ytick.color':       '#8b949e',
        'grid.color':        '#21262d',
        'grid.alpha':        0.6,
        'figure.dpi':        100,
    })


def get_psd_plot(raw: mne.io.Raw, title: str = "Power Spectral Density") -> plt.Figure:
    _dark_style()
    fig = raw.compute_psd(fmin=1, fmax=50, verbose=False).plot(show=False)
    fig.suptitle(title, color='#c9d1d9', fontsize=13, fontweight='bold')
    fig.patch.set_facecolor('#0d1117')
    for ax in fig.axes:
        ax.set_facecolor('#161b22')
    plt.tight_layout()
    return fig


def get_time_series_plot(
    raw: mne.io.Raw,
    duration: float = 5,
    title: str = "EEG Time Series",
) -> plt.Figure:
    _dark_style()
    fig = raw.plot(
        n_channels=len(raw.ch_names),
        duration=duration,
        show=False,
        scalings='auto',
        title=title,
        bgcolor='#161b22',
        color='#58a6ff',
    )
    fig.patch.set_facecolor('#0d1117')
    return fig


def get_topomap_fig(raw: mne.io.Raw, time_sec: float) -> plt.Figure:
    _dark_style()
    sfreq = raw.info['sfreq']
    idx   = int(time_sec * sfreq)
    idx   = max(0, min(idx, raw.n_times - 1))
    data  = raw.get_data(start=idx, stop=idx + 1).flatten()

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    mne.viz.plot_topomap(
        data, raw.info, axes=ax, show=False,
        cmap='RdBu_r', contours=4, sensors=True,
    )
    ax.set_title(f"Scalp Map @ {time_sec:.3f}s", color='#c9d1d9', fontsize=11)
    plt.tight_layout()
    return fig


def get_ica_components_fig(ica: mne.preprocessing.ICA, raw: mne.io.Raw) -> list[plt.Figure]:
    _dark_style()
    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]
    for f in figs:
        f.patch.set_facecolor('#0d1117')
    return figs


def get_all_channels_stacked_fig(
    raw: mne.io.Raw,
    start: float = 0.0,
    duration: float = 5.0,
    title: str = "EEG Signals",
    color: str = '#58a6ff',
) -> plt.Figure:
    """Plot ALL channels as vertically stacked subplots (like a classic EEG strip chart)."""
    _dark_style()

    n_ch = len(raw.ch_names)
    sfreq = raw.info['sfreq']
    start_idx = int(start * sfreq)
    stop_idx  = int((start + duration) * sfreq)
    stop_idx  = min(stop_idx, raw.n_times)

    data = raw.get_data(start=start_idx, stop=stop_idx) * 1e6   # V → µV
    n_samples = data.shape[1]
    times = np.linspace(start, start + duration, n_samples)

    fig_height = max(6, n_ch * 1.3)
    fig, axes = plt.subplots(
        n_ch, 1, figsize=(12, fig_height), facecolor='#0d1117',
        sharex=True, gridspec_kw={'hspace': 0.05},
    )
    if n_ch == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_facecolor('#0d1117')
        ax.plot(times, data[i], color=color, linewidth=0.5, alpha=0.85)
        ax.set_ylabel('µV', color='#8b949e', fontsize=7, labelpad=2)
        ax.set_title(
            raw.ch_names[i], color='#c9d1d9', fontsize=9,
            fontweight='bold', loc='left', pad=2,
        )
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.tick_params(axis='y', colors='#8b949e', labelsize=7, length=2)
        ax.tick_params(axis='x', colors='#8b949e', labelsize=7, length=2)
        ax.grid(True, alpha=0.1, color='#30363d')

    # Only show x-axis label on bottom subplot
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].set_xlabel('Time (s)', color='#8b949e', fontsize=9)

    fig.suptitle(
        title, color='#c9d1d9', fontsize=14, fontweight='bold',
        fontfamily='monospace', y=1.0,
    )
    plt.tight_layout()
    return fig


def get_band_power_bar(band_powers: dict) -> plt.Figure:
    _dark_style()
    bands  = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = [BAND_COLORS.get(b.split()[0], '#58a6ff') for b in bands]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    bars = ax.barh(bands, powers, color=colors, edgecolor='#30363d', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, powers):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%', va='center', color='#c9d1d9', fontsize=9,
        )
    ax.set_xlabel('Relative Power (%)', color='#8b949e')
    ax.set_title('Frequency Band Distribution', color='#c9d1d9', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', colors='#8b949e')
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────
def calculate_band_powers(raw: mne.io.Raw) -> dict[str, float]:
    """Return relative power (%) for each EEG frequency band."""
    psd = raw.compute_psd(fmin=1, fmax=45, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)
    avg_psd = psd_data.mean(axis=0)

    raw_powers = {}
    for band, (fmin, fmax) in FREQ_BANDS.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        raw_powers[band] = float(avg_psd[idx].mean()) if idx.any() else 0.0

    total = sum(raw_powers.values()) or 1.0
    return {b: (p / total) * 100 for b, p in raw_powers.items()}


def get_channel_stats(raw: mne.io.Raw) -> pd.DataFrame:
    """Per-channel descriptive statistics in µV."""
    data_uv = raw.get_data() * 1e6
    records = []
    for i, ch in enumerate(raw.ch_names):
        ch_data = data_uv[i]
        records.append({
            'Channel': ch,
            'Mean (µV)':   round(ch_data.mean(), 4),
            'Std (µV)':    round(ch_data.std(),  4),
            'Min (µV)':    round(ch_data.min(),  4),
            'Max (µV)':    round(ch_data.max(),  4),
            'RMS (µV)':    round(np.sqrt(np.mean(ch_data ** 2)), 4),
        })
    return pd.DataFrame(records)


def infer_cognitive_state(band_powers: dict[str, float]) -> tuple[str, str]:
    """Heuristic dominant-band cognitive label + emoji."""
    dominant = max(band_powers, key=band_powers.get)
    labels = {
        'Delta': ('Deep Sleep / Low Arousal', '😴'),
        'Theta': ('Drowsy / Meditative',       '🧘'),
        'Alpha': ('Relaxed Alertness',          '😌'),
        'Beta':  ('Active / Focused',           '🎯'),
        'Gamma': ('High Cognitive Load',        '⚡'),
    }
    label, emoji = labels.get(dominant, ('Unknown', '❓'))
    return label, emoji


# ──────────────────────────────────────────────
# EXPORT
# ──────────────────────────────────────────────
def export_clean_edf(raw: mne.io.Raw, output_path: str) -> str:
    """Export cleaned data as EDF. Returns path."""
    mne.export.export_raw(output_path, raw, fmt='edf', overwrite=True, verbose=False)
    return output_path


def export_excel(raw: mne.io.Raw, output_path: str, band_powers: dict | None = None) -> str:
    """Export time-series + stats + (optionally) band powers to Excel."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Time-series (sampled to keep size reasonable)
        df_ts = raw.to_data_frame()
        df_ts['time_ms'] = df_ts['time'] * 1e3
        eeg_cols = [c for c in df_ts.columns if c not in ['time', 'time_ms']]
        df_ts[eeg_cols] = df_ts[eeg_cols] * 1e6  # V → µV
        # Downsample: keep every Nth row so file stays < ~50 MB
        step = max(1, len(df_ts) // 50_000)
        df_ts[['time_ms'] + eeg_cols].iloc[::step].reset_index(drop=True).to_excel(
            writer, sheet_name='EEG_Timeseries', index=False
        )

        # Sheet 2: Channel statistics
        get_channel_stats(raw).to_excel(writer, sheet_name='Channel_Statistics', index=False)

        # Sheet 3: Band powers
        if band_powers:
            bp_df = pd.DataFrame([
                {
                    'Band': b,
                    'Relative Power (%)': round(p, 2),
                    'Description': BAND_DESCRIPTIONS.get(b, ''),
                }
                for b, p in band_powers.items()
            ])
            bp_df.to_excel(writer, sheet_name='Band_Powers', index=False)

    return output_path


def export_csv(raw: mne.io.Raw) -> bytes:
    """Return CSV bytes of the time-series."""
    df = raw.to_data_frame()
    df['time_ms'] = df['time'] * 1e3
    eeg_cols = [c for c in df.columns if c not in ['time', 'time_ms']]
    # df[eeg_cols] = df[eeg_cols] * 1e6
    print(df[eeg_cols])
    return df[['time_ms'] + eeg_cols].to_csv(index=False).encode('utf-8')
