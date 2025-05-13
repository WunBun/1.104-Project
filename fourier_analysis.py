import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load csv
# df = pd.read_csv('physical_data/physical_model_no_control_1.csv')  
df = pd.read_csv('FEM_timeseries_PID_0_0_0.csv')

# extract data from csv
t = df['t_s'].values
dt = t[1] - t[0] # sample interval
fs = 1.0 / dt # sample freq
n = len(t) # number of time samples
freqs = np.fft.rfftfreq(n, d=dt)

signal_cols = [col for col in df.columns if 'err' in col]

# compute ffts and plot
plt.figure(figsize=(10, 6))
for col in signal_cols:
    signal = df[col].fillna(0).values
    fft_result = np.fft.rfft(signal)
    amplitude = np.abs(fft_result) / n
    plt.plot(freqs, amplitude, label=col)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Controller Signals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# dominant frequencies
print("Dominant frequencies (peak FFT bin) for each controller:")
for col in signal_cols:
    signal = df[col].fillna(0).values
    fft_result = np.fft.rfft(signal)
    amplitude = np.abs(fft_result)
    peak_idx = np.argmax(amplitude)
    print(f"{col}: {freqs[peak_idx]:.3f} Hz")
