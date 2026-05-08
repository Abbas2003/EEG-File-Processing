"""
NeuroFlow — EEG Preprocessing & Analysis Dashboard
Run:  streamlit run app.py
"""

import os
import io
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline import (
    # loading
    load_raw_data,
    # preprocessing
    apply_filtering, run_ica_analysis, get_cleaned_data, full_pipeline,
    # visualisation
    get_psd_plot, get_time_series_plot, get_topomap_fig,
    get_ica_components_fig, get_band_power_bar,
    get_all_channels_stacked_fig,
    # analysis
    calculate_band_powers, get_channel_stats, infer_cognitive_state,
    # export
    export_clean_edf, export_excel, export_csv,
    # constants
    BAND_DESCRIPTIONS, BAND_COLORS,
)


# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroFlow · EEG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Import fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Root palette ---- */
:root {
    --bg:        #080c10;
    --surface:   #0d1117;
    --surface2:  #161b22;
    --border:    #21262d;
    --accent:    #58a6ff;
    --accent2:   #3fb950;
    --accent3:   #f78166;
    --accent4:   #d2a8ff;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

/* ---- Base overrides ---- */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ---- Header ---- */
.nf-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.nf-header::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(88,166,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.nf-header::after {
    content: '';
    position: absolute;
    bottom: -40px; right: 60px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(63,185,80,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.nf-logo {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -1px;
    position: relative; z-index: 1;
}
.nf-logo span { color: var(--accent); }
.nf-subtitle {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.3rem;
    position: relative; z-index: 1;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ---- Metric cards ---- */
.nf-metric {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.nf-metric:hover { border-color: var(--accent); }
.nf-metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: var(--mono);
    margin-bottom: 0.4rem;
}
.nf-metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent);
    font-family: var(--mono);
}
.nf-metric-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ---- Status badge ---- */
.nf-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: var(--mono);
    font-weight: 700;
    letter-spacing: 0.5px;
}
.nf-badge-raw   { background: rgba(210,168,255,0.12); color: var(--accent4); border: 1px solid rgba(210,168,255,0.3); }
.nf-badge-clean { background: rgba(63,185,80,0.12);  color: var(--accent2); border: 1px solid rgba(63,185,80,0.3); }

/* ---- Section headings ---- */
.nf-section {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

/* ---- File cards (sidebar) ---- */
.nf-file-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 6px;
    font-size: 0.78rem;
    color: var(--muted);
    font-family: var(--mono);
}
.nf-file-card.active {
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 10px !important;
    gap: 2px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
    padding: 6px 18px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--accent) !important;
    color: var(--bg) !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #1f6feb 100%) !important;
    color: #0d1117 !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--accent2) !important;
    border: 1px solid rgba(63,185,80,0.4) !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: rgba(63,185,80,0.1) !important;
    border-color: var(--accent2) !important;
}

/* ── Info / Warning / Error ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: var(--sans) !important;
    border: 1px solid var(--border) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Sliders ── */
[data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] [data-testid="stSelectbox"],
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Tables / dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Progress / spinner ── */
[data-testid="stStatusWidget"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Number inputs ── */
input[type="number"], input[type="text"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* ── Cognitive state card ── */
.nf-cog-card {
    background: linear-gradient(135deg, var(--surface2) 0%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
}
.nf-cog-emoji  { font-size: 3rem; margin-bottom: 0.4rem; }
.nf-cog-label  { font-family: var(--mono); font-size: 1.1rem; color: var(--text); font-weight: 700; }
.nf-cog-sub    { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(f"""
    <div class="nf-metric">
        <div class="nf-metric-label">{label}</div>
        <div class="nf-metric-value">{value}</div>
        {"<div class='nf-metric-sub'>"+sub+"</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section_head(text: str):
    st.markdown(f'<div class="nf-section">{text}</div>', unsafe_allow_html=True)


def badge(text: str, style: str = "raw"):
    dot = "●"
    st.markdown(f'<span class="nf-badge nf-badge-{style}">{dot} {text}</span>',
                unsafe_allow_html=True)


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for key in ['raw_dict', 'active_file', 'pipeline_results']:
    if key not in st.session_state:
        st.session_state[key] = {} if key != 'active_file' else None


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nf-header">
    <div class="nf-logo">Neuro<span>Flow</span></div>
    <div class="nf-subtitle">
        EEG Preprocessing &amp; Analysis Dashboard &nbsp;·&nbsp;
        Multi-file · ICA Artifact Removal · Export-Ready
    </div>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload EDF Files")
    uploaded_files = st.file_uploader(
        "Drop one or more .edf files",
        type=["edf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.raw_dict:
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    with st.spinner(f"Loading {uf.name}…"):
                        raw = load_raw_data(tmp_path)
                    st.session_state.raw_dict[uf.name] = {
                        "raw": raw, "tmp_path": tmp_path
                    }
                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")

    if st.session_state.raw_dict:
        st.markdown("---")
        st.markdown("### 🗂 Loaded Files")
        file_names = list(st.session_state.raw_dict.keys())
        active = st.radio(
            "Select file to analyse:",
            file_names,
            index=file_names.index(st.session_state.active_file)
            if st.session_state.active_file in file_names else 0,
            label_visibility="collapsed",
        )
        st.session_state.active_file = active

        if st.button("🗑 Clear All Files"):
            st.session_state.raw_dict = {}
            st.session_state.active_file = None
            st.session_state.pipeline_results = {}
            st.rerun()

    # ── Pipeline settings ──
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")
    l_freq       = st.number_input("Low cutoff (Hz)",    value=1.0,  step=0.5, min_value=0.1)
    h_freq       = st.number_input("High cutoff (Hz)",   value=50.0, step=1.0, min_value=1.0)
    notch_freq   = st.number_input("Notch freq (Hz)",    value=50.0, step=1.0, min_value=1.0)
    ica_thresh   = st.slider("ICA artifact threshold", 1.0, 5.0, 2.5, 0.1)
    ts_duration  = st.slider("Time-series preview (s)", 1, 30, 5)


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if not st.session_state.raw_dict:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;color:#8b949e;">
        <div style="font-size:3rem;margin-bottom:1rem;">🧠</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#e6edf3;">
            Upload one or more EDF files to begin
        </div>
        <div style="font-size:0.85rem;margin-top:0.5rem;">
            Use the sidebar on the left to import your EEG recordings
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── ACTIVE FILE ───────────────────────────────────────────────────────────────
af    = st.session_state.active_file
raw   = st.session_state.raw_dict[af]["raw"]
res   = st.session_state.pipeline_results.get(af, {})   # cleaned results if run
raw_c = res.get("raw_clean")
ica   = res.get("ica")

# ── FILE INFO STRIP ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    metric_card("FILE", Path(af).stem[:12] + ("…" if len(Path(af).stem) > 12 else ""), af)
with col2:
    metric_card("CHANNELS", str(len(raw.ch_names)))
with col3:
    metric_card("SAMPLE RATE", f"{int(raw.info['sfreq'])} Hz")
with col4:
    duration = raw.times[-1]
    metric_card("DURATION", f"{duration:.1f}s", f"{duration/60:.1f} min")
with col5:
    date_str = raw.info['meas_date'].strftime('%Y-%m-%d') if raw.info.get('meas_date') else "N/A"
    metric_card("RECORDED", date_str)

st.markdown("")

# ── STATUS BADGE ──────────────────────────────────────────────────────────────
if raw_c:
    badge("CLEANED DATA ACTIVE", "clean")
else:
    badge("RAW DATA — RUN PIPELINE TO CLEAN", "raw")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_raw, tab_pre, tab_res, tab_cog, tab_exp = st.tabs([
    "📶  Raw Signal",
    "🛠  Preprocessing",
    "✨  Results",
    "🧠  Cognitive Analysis",
    "📥  Export",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · RAW SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab_raw:
    section_head("Raw Signal Overview")

    col_psd, col_ts = st.columns(2)
    with col_psd:
        st.markdown("**Power Spectral Density**")
        with st.spinner("Computing PSD…"):
            st.pyplot(get_psd_plot(raw, "Raw — Power Spectral Density"), use_container_width=True)
    with col_ts:
        st.markdown("**Time Series**")
        with st.spinner("Rendering signals…"):
            st.pyplot(get_time_series_plot(raw, duration=ts_duration, title="Raw — Time Series"),
                      use_container_width=True)

    section_head("Channel Information")
    ch_info = pd.DataFrame({
        "Channel":  raw.ch_names,
        "Type":     [raw.get_channel_types()[i] for i in range(len(raw.ch_names))],
    })
    st.dataframe(ch_info, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
with tab_pre:
    section_head("Preprocessing Pipeline")

    st.markdown("""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
    <b style="color:#58a6ff;font-family:'Space Mono',monospace;">Pipeline steps</b><br><br>
    <span style="color:#8b949e;font-size:0.88rem;">
    1 · Bandpass filter &nbsp;(configurable low / high cutoff)<br>
    2 · Notch filter &nbsp;(removes power-line interference)<br>
    3 · ICA decomposition &nbsp;(FastICA)<br>
    4 · Ocular artifact auto-detection &amp; removal<br>
    5 · Average re-referencing
    </span>
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run Pipeline", use_container_width=True)

    if run_btn:
        with st.status("Running preprocessing pipeline…", expanded=True) as status:
            st.write(f"⚙️  Bandpass filter: {l_freq}–{h_freq} Hz")
            raw_f = apply_filtering(raw, l_freq, h_freq, notch_freq)

            st.write(f"⚙️  Notch filter: {notch_freq} Hz")

            st.write("⚙️  Fitting ICA — this may take a moment…")
            ica_result = run_ica_analysis(raw_f, threshold=ica_thresh)

            n_exc = len(ica_result.exclude)
            st.write(f"✅  Detected {n_exc} artifact component{'s' if n_exc != 1 else ''}")

            st.write("⚙️  Applying ICA & average re-reference…")
            raw_clean_result = get_cleaned_data(raw_f, ica_result)

            st.session_state.pipeline_results[af] = {
                "raw_clean": raw_clean_result,
                "ica":       ica_result,
            }
            status.update(label="✅ Pipeline complete!", state="complete")

        st.success("Head to **Results** and **Cognitive Analysis** tabs to explore the cleaned data.")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_res:
    if not raw_c:
        st.info("Run the preprocessing pipeline first to see results here.")
    else:
        section_head("ICA Artifact Removal")
        n_exc = len(ica.exclude)
        st.markdown(
            f'<span style="color:#8b949e;font-size:0.9rem;">Removed '
            f'<b style="color:#f78166;">{n_exc}</b> artifact component(s) via ICA.</span>',
            unsafe_allow_html=True,
        )

        with st.expander("🔍 View ICA Components", expanded=False):
            with st.spinner("Rendering components…"):
                figs = get_ica_components_fig(ica, raw_c)
                for f in figs:
                    st.pyplot(f, use_container_width=True)

        section_head("Cleaned Signal Comparison")
        col_psd2, col_ts2 = st.columns(2)
        with col_psd2:
            st.markdown("**Cleaned PSD**")
            st.pyplot(get_psd_plot(raw_c, "Cleaned — Power Spectral Density"),
                      use_container_width=True)
        with col_ts2:
            st.markdown("**Cleaned Time Series**")
            st.pyplot(get_time_series_plot(raw_c, duration=ts_duration, title="Cleaned — Time Series"),
                      use_container_width=True)

        section_head("All Channels — Before vs After Processing")
        st.markdown(
            '<span style="color:#8b949e;font-size:0.88rem;">'
            'Compare every channel at once. <b style="color:#d2a8ff;">Purple = Raw</b> '
            '&nbsp;·&nbsp; <b style="color:#3fb950;">Green = Cleaned</b>'
            '</span>',
            unsafe_allow_html=True,
        )

        max_t = float(raw.times[-1])
        col_start, col_dur = st.columns(2)
        with col_start:
            comp_start = st.slider(
                "Start time (s)", 0.0, max(0.0, max_t - 1.0), 0.0,
                step=0.5, key="comp_all_start",
            )
        with col_dur:
            comp_dur = st.slider(
                "Duration (s)", 1.0, min(30.0, max_t - comp_start), 5.0,
                step=0.5, key="comp_all_dur",
            )

        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown(
                '<div style="text-align:center;padding:6px 0;">'
                '<span class="nf-badge nf-badge-raw">● BEFORE PROCESSING</span></div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Rendering raw signals…"):
                fig_raw = get_all_channels_stacked_fig(
                    raw, comp_start, comp_dur,
                    title="Raw Signal — All Channels",
                    color='#d2a8ff',
                )
                st.pyplot(fig_raw, use_container_width=True)
                plt.close(fig_raw)

        with col_after:
            st.markdown(
                '<div style="text-align:center;padding:6px 0;">'
                '<span class="nf-badge nf-badge-clean">● AFTER PROCESSING</span></div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Rendering cleaned signals…"):
                fig_clean = get_all_channels_stacked_fig(
                    raw_c, comp_start, comp_dur,
                    title="Cleaned Signal — All Channels",
                    color='#3fb950',
                )
                st.pyplot(fig_clean, use_container_width=True)
                plt.close(fig_clean)

        section_head("Channel Statistics")
        with st.spinner("Computing statistics…"):
            stats_df = get_channel_stats(raw_c)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · COGNITIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_cog:
    data_for_cog = raw_c if raw_c else raw

    if not raw_c:
        st.warning("For accurate analysis, run the pipeline first. Showing **raw** data below.")

    section_head("Frequency Band Analysis")

    with st.spinner("Computing band powers…"):
        bands = calculate_band_powers(data_for_cog)
    cog_label, cog_emoji = infer_cognitive_state(bands)

    col_cog, col_bands = st.columns([1, 2])

    with col_cog:
        st.markdown(f"""
        <div class="nf-cog-card">
            <div class="nf-cog-emoji">{cog_emoji}</div>
            <div class="nf-cog-label">{cog_label}</div>
            <div class="nf-cog-sub">Dominant cognitive state</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        band_df = pd.DataFrame([
            {"Band": b, "Power (%)": f"{p:.1f}%"}
            for b, p in bands.items()
        ])
        st.dataframe(band_df, hide_index=True, use_container_width=True)

    with col_bands:
        st.pyplot(get_band_power_bar(bands), use_container_width=True)

    with st.expander("📖 Band Interpretations"):
        for band, desc in BAND_DESCRIPTIONS.items():
            pct = bands.get(band, 0)
            color = BAND_COLORS.get(band, "#58a6ff")
            st.markdown(
                f'<span style="color:{color};font-weight:700;font-family:monospace;">'
                f'{band} ({pct:.1f}%)</span>'
                f'<span style="color:#8b949e;"> — {desc}</span>',
                unsafe_allow_html=True,
            )

    section_head("Scalp Topographic Map")
    max_t = float(data_for_cog.times[-1])
    t_sel = st.slider("Time point (seconds)", 0.0, max_t, 0.0, step=0.01)
    col_topo, col_topo_info = st.columns([1, 1])
    with col_topo:
        with st.spinner("Rendering topomap…"):
            st.pyplot(get_topomap_fig(data_for_cog, t_sel), use_container_width=True)
    with col_topo_info:
        st.markdown("""
        <div style="color:#8b949e;font-size:0.85rem;padding:1rem 0;">
        <b style="color:#e6edf3;">How to read the topomap</b><br><br>
        🔴 <b>Red</b> — positive scalp potential<br>
        🔵 <b>Blue</b> — negative scalp potential<br><br>
        The dots represent electrode positions.
        Contour lines show regions of equal potential.
        Use the slider to scan through the recording.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_exp:
    section_head("Export Cleaned Data")

    if not raw_c:
        st.warning("Run the preprocessing pipeline first to enable export of cleaned data.")
        st.markdown("You can still export the **raw** data below.")

    data_to_export = raw_c if raw_c else raw
    label = "cleaned" if raw_c else "raw"
    stem  = Path(af).stem

    # ── Row 1: EDF + Excel ──
    col_edf, col_xlsx, col_csv = st.columns(3)

    with col_edf:
        st.markdown(f"""
        <div class="nf-metric" style="text-align:left;">
            <div class="nf-metric-label">EDF Export</div>
            <div style="font-size:0.85rem;color:#8b949e;margin:0.4rem 0;">
                Clean EDF file ready for<br>downstream analysis tools.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Generate EDF", key="gen_edf", use_container_width=True):
            with st.spinner("Writing EDF file…"):
                tmp_edf = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
                tmp_edf.close()
                export_clean_edf(data_to_export, tmp_edf.name)
                with open(tmp_edf.name, "rb") as f:
                    edf_bytes = f.read()
                st.session_state[f"edf_bytes_{af}"] = edf_bytes

        if f"edf_bytes_{af}" in st.session_state:
            st.download_button(
                label=f"📥 Download {stem}_{label}.edf",
                data=st.session_state[f"edf_bytes_{af}"],
                file_name=f"{stem}_{label}.edf",
                mime="application/octet-stream",
                use_container_width=True,
            )

    with col_xlsx:
        st.markdown(f"""
        <div class="nf-metric" style="text-align:left;">
            <div class="nf-metric-label">Excel Export</div>
            <div style="font-size:0.85rem;color:#8b949e;margin:0.4rem 0;">
                3 sheets: time-series, channel<br>stats &amp; band powers.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Generate Excel", key="gen_xlsx", use_container_width=True):
            with st.spinner("Building Excel workbook…"):
                bp = calculate_band_powers(data_to_export)
                tmp_xl = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
                tmp_xl.close()
                export_excel(data_to_export, tmp_xl.name, band_powers=bp)
                with open(tmp_xl.name, "rb") as f:
                    xlsx_bytes = f.read()
                st.session_state[f"xlsx_bytes_{af}"] = xlsx_bytes

        if f"xlsx_bytes_{af}" in st.session_state:
            st.download_button(
                label=f"📥 Download {stem}_{label}.xlsx",
                data=st.session_state[f"xlsx_bytes_{af}"],
                file_name=f"{stem}_{label}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col_csv:
        st.markdown(f"""
        <div class="nf-metric" style="text-align:left;">
            <div class="nf-metric-label">CSV Export</div>
            <div style="font-size:0.85rem;color:#8b949e;margin:0.4rem 0;">
                Flat CSV with time + all<br>channel values in µV.
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Preparing CSV…"):
            csv_bytes = export_csv(data_to_export)
        st.download_button(
            label=f"📥 Download {stem}_{label}.csv",
            data=csv_bytes,
            file_name=f"{stem}_{label}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Plot export ──
    section_head("Export Plots (PNG)")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        if st.button("🖼 PSD Plot", use_container_width=True):
            psd_fig = get_psd_plot(data_to_export, f"{stem} — PSD")
            st.session_state[f"psd_png_{af}"] = fig_to_bytes(psd_fig)
        if f"psd_png_{af}" in st.session_state:
            st.download_button(
                "📥 psd.png", st.session_state[f"psd_png_{af}"],
                f"{stem}_psd.png", "image/png", use_container_width=True,
            )

    with col_p2:
        if st.button("🖼 Time Series Plot", use_container_width=True):
            ts_fig = get_time_series_plot(data_to_export, ts_duration)
            st.session_state[f"ts_png_{af}"] = fig_to_bytes(ts_fig)
        if f"ts_png_{af}" in st.session_state:
            st.download_button(
                "📥 timeseries.png", st.session_state[f"ts_png_{af}"],
                f"{stem}_timeseries.png", "image/png", use_container_width=True,
            )

    with col_p3:
        if st.button("🖼 Topomap Plot", use_container_width=True):
            topo_fig = get_topomap_fig(data_to_export, 0.0)
            st.session_state[f"topo_png_{af}"] = fig_to_bytes(topo_fig)
        if f"topo_png_{af}" in st.session_state:
            st.download_button(
                "📥 topomap.png", st.session_state[f"topo_png_{af}"],
                f"{stem}_topomap.png", "image/png", use_container_width=True,
            )

    # ── Multi-file batch export ──
    if len(st.session_state.raw_dict) > 1:
        section_head("Batch Export (All Files)")
        n_ready = sum(1 for k in st.session_state.raw_dict if k in st.session_state.pipeline_results)
        st.markdown(
            f'<span style="color:#8b949e;font-size:0.88rem;">'
            f'{n_ready} / {len(st.session_state.raw_dict)} files have been processed.</span>',
            unsafe_allow_html=True,
        )

        if st.button("📦 Batch Export All CSVs as ZIP", use_container_width=False):
            import zipfile, io as _io
            zip_buf = _io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, fdata in st.session_state.raw_dict.items():
                    results = st.session_state.pipeline_results.get(fname, {})
                    r = results.get("raw_clean", fdata["raw"])
                    lbl = "cleaned" if "raw_clean" in results else "raw"
                    csv_b = export_csv(r)
                    zf.writestr(f"{Path(fname).stem}_{lbl}.csv", csv_b)
            zip_buf.seek(0)
            st.download_button(
                "📥 Download all_files.zip",
                zip_buf.read(),
                "neuroflow_export.zip",
                "application/zip",
            )
