import os
import re
from datetime import datetime, timedelta
import numpy as np

# ─────────────────────────────────────────────
# SHARED CONFIGURATION
# ─────────────────────────────────────────────
ROOT_DIR       = 'siena'
CACHE_DIR      = 'siena_stft_cache'
CHECKPOINT_DIR = 'checkpoints'

ALL_PATIENTS = np.array([
    'PN00', 'PN01', 'PN03', 'PN05', 'PN06', 'PN07',
    'PN09', 'PN10', 'PN11', 'PN12', 'PN13', 'PN14', 'PN16', 'PN17'
])

TEST_PATIENTS = ['PN01', 'PN07', 'PN11']
CV_PATIENTS   = np.array([p for p in ALL_PATIENTS if p not in TEST_PATIENTS])

EPOCH_DURATION   = 2.0
OVERLAP_DURATION = 1.0

STANDARD_CHANNELS = [
    'EEG Fp1', 'EEG F3',  'EEG C3',  'EEG P3',
    'EEG F7',  'EEG T3',  'EEG T5',  'EEG Fz',
    'EEG Cz',  'EEG Pz',  'EEG F4',  'EEG C4',
    'EEG P4',  'EEG O2',  'EEG F8',  'EEG T4',
    'EEG T6',  'EEG O1',
]

# Parameters for STFT computation
N_FFT      = 256
HOP_LENGTH = 128
SFREQ      = 512

# Derive frequency bin mask
_freqs     = np.fft.rfftfreq(N_FFT, d=1.0 / SFREQ)
FREQ_MASK  = (_freqs >= 1.0) & (_freqs <= 40.0)

# Evaluates to 20 bins for 1.0 - 40.0 Hz at SFREQ=512 and N_FFT=256
N_FREQ_BINS = int(FREQ_MASK.sum()) 

# ─────────────────────────────────────────────
# TIME PARSING
# ─────────────────────────────────────────────
def parse_time(time_str):
    time_str = time_str.strip().replace(':', '.').replace(' ', '')
    parts    = time_str.split('.')
    h, m, s  = int(parts[0]), int(parts[1]), int(parts[2])
    return datetime(2000, 1, 1, h, m, s)

def compute_offset_seconds(reg_start_str, event_time_str):
    reg_start  = parse_time(reg_start_str)
    event_time = parse_time(event_time_str)
    if event_time < reg_start:
        event_time += timedelta(hours=24)
    return (event_time - reg_start).total_seconds()

def parse_seizure_list(txt_path):
    if not os.path.exists(txt_path):
        return {}
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    seizures = {}
    blocks   = re.split(r'Seizure n\s*\d+', content, flags=re.IGNORECASE)
    blocks   = [b.strip() for b in blocks if b.strip()]

    for block in blocks:
        lines     = [l.strip() for l in block.split('\n') if l.strip()]
        file_name = None

        for line in lines:
            if line.lower().startswith('file name'):
                raw_name   = line.split(':', 1)[-1].strip()
                raw_name   = re.sub(r'\(.*?\)', '', raw_name).strip()
                file_name  = raw_name.replace('PNO', 'PN0')
                break

        if file_name is None:
            continue

        reg_start = sz_start = sz_end = None
        for line in lines:
            if 'registration start time' in line.lower():
                m = re.search(r'\d{1,2}[.:]\d{2}[.:]\d{2}', line.split(':', 1)[-1].strip())
                if m: reg_start = m.group()
            if 'seizure start time' in line.lower():
                m = re.search(r'\d{1,2}[.:]\d{2}[.:]\d{2}', line.split(':', 1)[-1].strip())
                if m: sz_start = m.group()
            if 'seizure end time' in line.lower():
                m = re.search(r'\d{1,2}[.:]\d{2}[.:]\d{2}', line.split(':', 1)[-1].strip())
                if m: sz_end = m.group()

        if reg_start is None or sz_start is None or sz_end is None:
            continue

        try:
            start_sec = compute_offset_seconds(reg_start, sz_start)
            end_sec   = compute_offset_seconds(reg_start, sz_end)
        except Exception:
            continue

        if file_name not in seizures:
            seizures[file_name] = []
        seizures[file_name].append((start_sec, end_sec))

    return seizures

def setup_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
