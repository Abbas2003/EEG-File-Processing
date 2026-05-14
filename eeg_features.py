import numpy as np
import pandas as pd
import mne
# pyrefly: ignore [missing-import]
import antropy as ant
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Frequency Bands
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 45)
}

def extract_all_features(epochs: mne.Epochs) -> pd.DataFrame:
    """
    Extract ~370 features from each epoch.
    Returns a DataFrame where each row is an epoch.
    """
    sfreq = epochs.info['sfreq']
    ch_names = epochs.ch_names
    data = epochs.get_data(copy=True)  # (n_epochs, n_channels, n_samples)
    
    all_epoch_features = []
    
    for e_idx in range(len(epochs)):
        epoch_data = data[e_idx] * 1e6  # V to µV
        feat_dict = {
            'epoch_id': e_idx,
            'time_start': epochs.events[e_idx, 0] / sfreq,
            'time_end': (epochs.events[e_idx, 0] / sfreq) + (epochs.selection.shape[0] / sfreq if hasattr(epochs, 'selection') else 4.0)
        }
        # Actually time_end is just time_start + 4.0 (the duration)
        feat_dict['time_end'] = feat_dict['time_start'] + (epochs.tmax - epochs.tmin)
        
        # 1. PER-CHANNEL FEATURES
        for c_idx, ch in enumerate(ch_names):
            x = epoch_data[c_idx]
            
            # --- Time Domain ---
            feat_dict[f'{ch}_mean'] = np.mean(x)
            feat_dict[f'{ch}_std']  = np.std(x)
            feat_dict[f'{ch}_var']  = np.var(x)
            feat_dict[f'{ch}_skew'] = skew(x)
            feat_dict[f'{ch}_kurt'] = kurtosis(x)
            feat_dict[f'{ch}_rms']  = np.sqrt(np.mean(x**2))
            # Some antropy versions use num_zerocrossings, others use num_zerocross
            if hasattr(ant, 'num_zerocross'):
                feat_dict[f'{ch}_zcr'] = ant.num_zerocross(x)
            else:
                feat_dict[f'{ch}_zcr'] = ant.num_zerocrossings(x)
            
            # Hjorth
            if hasattr(ant, 'hjorth_params'):
                mob, com = ant.hjorth_params(x)
            else:
                mob = ant.hjorth_mobility(x)
                com = ant.hjorth_complexity(x)
                
            feat_dict[f'{ch}_hjorth_activity']   = np.var(x)
            feat_dict[f'{ch}_hjorth_mobility']   = mob
            feat_dict[f'{ch}_hjorth_complexity'] = com
            
            # Entropy
            feat_dict[f'{ch}_spec_entropy'] = ant.spectral_entropy(x, sfreq, method='welch', normalize=True)
            
            if hasattr(ant, 'perm_entropy'):
                feat_dict[f'{ch}_perm_entropy'] = ant.perm_entropy(x, normalize=True)
            else:
                feat_dict[f'{ch}_perm_entropy'] = ant.permutation_entropy(x, normalize=True)
            feat_dict[f'{ch}_higuchi_fd']   = ant.higuchi_fd(x)
            
            # --- Frequency Domain ---
            psd, freqs = welch(x, sfreq, nperseg=int(sfreq*2))
            
            band_powers = {}
            for band, (fmin, fmax) in FREQ_BANDS.items():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                bp = np.mean(psd[idx]) if np.any(idx) else 0.0
                band_powers[band] = bp
                feat_dict[f'{ch}_{band}_abs'] = bp
            
            total_power = sum(band_powers.values()) or 1e-10
            for band, bp in band_powers.items():
                feat_dict[f'{ch}_{band}_rel'] = (bp / total_power) * 100
                
            # Ratios per channel
            feat_dict[f'{ch}_theta_beta_ratio'] = band_powers['theta'] / (band_powers['beta'] or 1e-10)
            feat_dict[f'{ch}_alpha_beta_ratio'] = band_powers['alpha'] / (band_powers['beta'] or 1e-10)
            feat_dict[f'{ch}_theta_alpha_ratio'] = band_powers['theta'] / (band_powers['alpha'] or 1e-10)

        # 2. CROSS-CHANNEL / GLOBAL FEATURES
        # Case-insensitive helper for asymmetry
        def get_asymmetry(ch_l, ch_r, band):
            # Find the actual channel name in feat_dict keys
            key_l = next((k for k in feat_dict.keys() if ch_l.lower() in k.lower() and band in k), None)
            key_r = next((k for k in feat_dict.keys() if ch_r.lower() in k.lower() and band in k), None)
            
            l_pow = feat_dict.get(key_l, 0) if key_l else 0
            r_pow = feat_dict.get(key_r, 0) if key_r else 0
            
            if l_pow > 0 and r_pow > 0:
                return np.log(r_pow) - np.log(l_pow)
            return 0.0

        feat_dict['faa_af_alpha'] = get_asymmetry('AF3', 'AF4', 'alpha')
        feat_dict['faa_f_alpha']  = get_asymmetry('F3', 'F4', 'alpha')
        feat_dict['faa_f_beta']   = get_asymmetry('F3', 'F4', 'beta')
        
        # 3. GLOBAL (AVERAGE) FEATURES & RELATIVE POWERS
        # Needed for clustering interpretation logic
        all_abs_powers = []
        for band in FREQ_BANDS.keys():
            b_vals = [feat_dict[f'{ch}_{band}_abs'] for ch in ch_names]
            avg_b = np.mean(b_vals)
            feat_dict[f'global_{band}_abs'] = avg_b
            all_abs_powers.append(avg_b)
            
        total_global_pow = sum(all_abs_powers) or 1e-10
        for band in FREQ_BANDS.keys():
            feat_dict[f'global_{band}_rel'] = feat_dict[f'global_{band}_abs'] / total_global_pow

        # Engagement & Ratios (Global)
        g_theta = feat_dict['global_theta_abs']
        g_alpha = feat_dict['global_alpha_abs']
        g_beta  = feat_dict['global_beta_abs']
        
        feat_dict['global_theta_beta_ratio'] = g_theta / (g_beta or 1e-10)
        feat_dict['global_theta_alpha_ratio'] = g_theta / (g_alpha or 1e-10)
        feat_dict['engagement_index'] = g_beta / ((g_alpha + g_theta) or 1e-10)
        
        # Stability fix: cap extreme ratios caused by noise/zeros
        for k, v in feat_dict.items():
            if 'ratio' in k or 'index' in k:
                if v > 1000: feat_dict[k] = 1000.0
        
        all_epoch_features.append(feat_dict)
        
    df = pd.DataFrame(all_epoch_features)
    # Handle any NaNs or Infinities that might have leaked from entropy/FD calculations
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple min-max scaling for UI preview, keep metadata columns."""
    meta_cols = ['epoch_id', 'time_start', 'time_end']
    cols_to_scale = [c for c in df.columns if c not in meta_cols]
    
    df_scaled = df.copy()
    for col in cols_to_scale:
        c_min = df[col].min()
        c_max = df[col].max()
        if c_max > c_min:
            df_scaled[col] = (df[col] - c_min) / (c_max - c_min)
        else:
            df_scaled[col] = 0.0
            
    return df_scaled
