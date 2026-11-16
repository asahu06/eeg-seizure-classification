import os
import numpy as np
import mne
import warnings
from utils import (
    ROOT_DIR, CACHE_DIR, STANDARD_CHANNELS, SFREQ, EPOCH_DURATION, 
    OVERLAP_DURATION, N_FFT, HOP_LENGTH, FREQ_MASK, parse_seizure_list, setup_dirs, setup_dirs
)

mne.set_log_level('ERROR')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')

def compute_stft_for_patient(patient_id, patient_dir, cache_path):
    """Computes STFT from raw EDF files and saves to disk cache."""
    print(f"  Computing STFT for {patient_id}...")
    txt_path    = os.path.join(patient_dir, f'Seizures-list-{patient_id}.txt')
    seizure_map = parse_seizure_list(txt_path)

    patient_X, patient_y = [], []
    edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf')])

    for edf_file in edf_files:
        file_path = os.path.join(patient_dir, edf_file)
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        except Exception as e:
            print(f"    {edf_file}: failed to open ({e}), skipping.")
            continue

        available_lower = {ch.lower(): ch for ch in raw.ch_names}
        pick_names      = []
        missing_ch      = None

        for ch in STANDARD_CHANNELS:
            if ch.lower() in available_lower:
                pick_names.append(available_lower[ch.lower()])
            else:
                missing_ch = ch
                break

        if missing_ch is not None:
            print(f"    {edf_file}: skipped — missing channel '{missing_ch}'.")
            raw.close()
            continue

        raw.pick_channels(pick_names, ordered=True)
        
        if raw.info['sfreq'] != SFREQ:
            raw.resample(SFREQ, verbose=False)

        raw.notch_filter(freqs=50.0, verbose=False)
        raw.filter(1.0, 40.0, method='iir', iir_params={'order': 4, 'ftype': 'butter'}, phase='zero', verbose=False)

        sfreq  = raw.info['sfreq']
        events = mne.make_fixed_length_events(raw, id=1, duration=EPOCH_DURATION, overlap=OVERLAP_DURATION)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_DURATION, baseline=None, preload=True, verbose=False)

        sz_intervals = seizure_map.get(edf_file, [])
        if not sz_intervals:
            for key in seizure_map:
                if key.lower() == edf_file.lower():
                    sz_intervals = seizure_map[key]
                    break

        epochs_data = epochs.get_data(verbose=False)
        n_epochs    = epochs_data.shape[0]
        window_step = EPOCH_DURATION - OVERLAP_DURATION

        for idx in range(n_epochs):
            onset_time = epochs.events[idx][0] / sfreq
            window_end = onset_time + EPOCH_DURATION
            is_seizure = 0
            
            for sz_start, sz_end in sz_intervals:
                if onset_time < sz_end and window_end > sz_start:
                    is_seizure = 1
                    break

            stft_complex   = mne.time_frequency.stft(epochs_data[idx], wsize=N_FFT, tstep=HOP_LENGTH, verbose=False)
            stft_magnitude = np.abs(stft_complex)[:, FREQ_MASK, :]
            stft_grid      = np.log1p(np.transpose(stft_magnitude, (0, 2, 1))).astype(np.float32)

            patient_X.append(stft_grid)
            patient_y.append(is_seizure)

        raw.close()
        
        file_windows_added = len(epochs.events)
        file_y_labels = patient_y[-file_windows_added:] if file_windows_added > 0 else []
        file_seizures = sum(1 for l in file_y_labels if l == 1)
        print(f"    {edf_file}: {file_windows_added} windows ({file_seizures} seizures in this file)")

    if not patient_X:
        np.savez_compressed(cache_path, X=np.empty((0,)), y=np.empty((0,)))
        return None, None

    X_arr, y_arr = np.array(patient_X, dtype=np.float32), np.array(patient_y, dtype=np.float32)
    np.savez_compressed(cache_path, X=X_arr, y=y_arr)
    
    total_sz = int(y_arr.sum())
    print(f"  {patient_id}: {len(y_arr)} windows total ({total_sz} seizure, {len(y_arr)-total_sz} non-seizure)")
    return X_arr, y_arr

def get_patient_stft_data(patient_id):
    """Loads STFT data from cache or computes it if missing."""
    cache_path = os.path.join(CACHE_DIR, f"{patient_id}_stft.npz")

    if os.path.exists(cache_path):
        archive = np.load(cache_path)
        if len(archive['y']) == 0:
            return None, None
        return archive['X'], archive['y']

    patient_dir = os.path.join(ROOT_DIR, patient_id)
    if not os.path.isdir(patient_dir):
        return None, None

    return compute_stft_for_patient(patient_id, patient_dir, cache_path)

if __name__ == "__main__":
    from utils import ALL_PATIENTS
    setup_dirs()
    print("Populating cache for all patients...")
    for p in ALL_PATIENTS:
        get_patient_stft_data(p)
    print("Pre-processing complete.")