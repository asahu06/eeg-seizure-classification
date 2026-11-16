import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import torch
torch.__version__

# Load raw EEG data
file_path = "chbmit/chb01/chb01_03.edf"
raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

   # raw.pick(['T7-P7'])

    # 3. Crop a 20-second window right where the seizure starts (2996 seconds)
    # Let's grab 10 seconds before it starts, and 10 seconds after
    # start_time = 2986
    # end_time = 3046
    # raw.load_data().crop(tmin=start_time, tmax=end_time)

    # # 4. Extract the raw numbers and the time steps
    # data, times = raw[:, :]

    # # 5. Plot a simple 2D line graph
    # plt.figure(figsize=(12, 4))
    # plt.plot(times + start_time, data[0], color='black', linewidth=1)

    # # Draw a red vertical line at exactly 2996s so you know where the seizure hits
    # plt.axvline(x=2996, color='red', linestyle='--', label='Seizure Start (from summary text)')
    # plt.axvline(x=3036, color='red', linestyle='--', label='Seizure End (from summary text)')

    # plt.title("Visualizing Seizure Onset on Channel T7-P7")
    # plt.xlabel("Time (Seconds)")
    # plt.ylabel("Voltage")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

# Option 2: Print the full system information (includes sampling rate, etc.)
print("\n--- FULL RAW INFO ---")
print(raw.info)
print(raw.ch_names)