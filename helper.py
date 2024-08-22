
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib import animation

def compute_melspectrogram(audio_file):
    audio_data, sr = librosa.load(audio_file, sr=None)
    melspectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    return melspectrogram, sr

def plot_spectrogram(audio_data, sr):
    # Calculate the spectrogram
    spec = np.abs(librosa.stft(audio_data))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    return plt

def plot_waveform(y, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=y, sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title="Waveform")
    plt.xlabel(title="Time (s)")
    plt.ylabel(title="Amplitude")
    return plt

def compute_stft(y, sr):
    # Compute Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def compute_chroma(y, sr):
    # Compute chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma

def compute_mel(y, sr):
    # Compute mel spectrogram
    M = librosa.feature.melspectrogram(y=y, sr=sr)
    M_db = librosa.amplitude_to_db(M, ref=np.max)
    return M_db

def plot_chromagram(chromagram, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(chromagram, sr=sr, x_axis="time", y_axis="chroma", vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Real Audio Chromagram")
    return plt

def plot_mfccs(mfccs, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.colorbar()
    plt.title("Real Audio Mel-Frequency Cepstral Coefficients (MFCCs)")
    return plt

def plot_rms_curve(y, sr):
    # Compute RMS
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)))

    # Plot RMS curve
    fig, ax = plt.subplots()
    ax.plot(times, rms)
    ax.axhline(0.02, color='r', alpha=0.5)
    ax.set(xlabel='Time', ylabel='RMS')
    return fig


def plot_probability_curve(y, sr):
    # Compute RMS
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)))

    # Calculate probability
    r_normalized = (rms - 0.02) / np.std(rms)
    p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

    # Plot probability curve
    fig, ax = plt.subplots()
    ax.plot(times, p, label='P[V=1|x]')
    ax.axhline(0.5, color='r', alpha=0.5, label='Decision threshold')
    ax.set(xlabel='Time')
    ax.legend()
    return fig

def plot_spectrogram_with_threshold(y, sr):
    # Compute RMS
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)))

    # Compute full spectrogram
    S_full = np.abs(librosa.stft(y))

    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', x_axis='time', sr=sr, ax=ax[0])
    ax[0].label_outer()
    ax[1].step(times, rms >= 0.02, label='Non-silent')
    ax[1].set(ylim=[0, 4.05])
    ax[1].legend()
    return fig

def generate_spectrogram_animation(patches):
    fig, ax = plt.subplots()
    mesh = librosa.display.specshow(patches[..., 0], x_axis='time',
                                    y_axis='mel', ax=ax)

    def _update(num):
        mesh.set_array(patches[..., num])
        return (mesh,)

    ani = animation.FuncAnimation(fig,
                                  func=_update,
                                  frames=patches.shape[-1],
                                  interval=100,  # 100 milliseconds = 1/10 sec
                                  blit=True)
    
    return ani

def plot_power_spectral_density(y, sr):
    plt.figure(figsize=(14, 5))
    plt.psd(y, NFFT=2048, Fs=sr)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    return plt

def plot_cepstrum(y, sr):
    # Compute the cepstrum
    spectrum = np.fft.fft(y)
    log_spectrum = np.log(np.abs(spectrum))
    cepstrum = np.fft.ifft(log_spectrum).real

    plt.figure(figsize=(14, 5))
    plt.plot(cepstrum)
    plt.title("Cepstrum")
    plt.xlabel("Quefrency (samples)")
    plt.ylabel("Amplitude")
    return plt

def plot_wavelet_transform(y, sr):
    try:
        import pywt
        # Compute wavelet transform
        coeffs = pywt.wavedec(y, 'haar', level=6)
        plt.figure(figsize=(14, 5))
        for i, coeff in enumerate(coeffs):
            plt.plot(coeff, label=f'Level {i}')
        plt.title("Wavelet Transform")
        plt.xlabel("Sample Index")
        plt.ylabel("Coefficient Value")
        plt.legend()
        return plt
    except ModuleNotFoundError as e:
        print(f"{e} Module not found!!!")


def plot_tonal_analysis(y, sr):
    # Compute tonal analysis features
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(tonnetz, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("Tonal Analysis")
    return plt

def plot_timbre_features(y, sr):
    # Example: Spectral Flatness and Bandwidth
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 1, 1)
    plt.plot(spectral_flatness[0])
    plt.title("Spectral Flatness")
    plt.xlabel("Time")
    plt.ylabel("Spectral Flatness")
    
    plt.subplot(2, 1, 2)
    plt.plot(spectral_bandwidth[0])
    plt.title("Spectral Bandwidth")
    plt.xlabel("Time")
    plt.ylabel("Spectral Bandwidth")
    plt.tight_layout()
    return plt
