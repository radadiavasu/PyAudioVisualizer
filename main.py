import streamlit as st
from helper import *
import librosa
import matplotlib.pyplot as plt
import numpy as np
import wave
import tempfile
import io


def main():

    html_content = """
    <div class="version">
    <div class="demo version-section">
        <a href="https://github.com/radadiavasu" class="github-corner">
        <svg width="80" height="80" viewBox="0 0 250 250" style="fill:#151513; color:#fff; position: absolute; top: -100; border: 0; left: 0; transform: scale(-1, 1);">
            <path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path>
            <path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path>
            <path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path>
        </svg>
        </a>
    </div>
    </div>
    """

    css_content = """
    .github-corner:hover .octo-arm {
    animation: octocat-wave 560ms ease-in-out;
    }

    @keyframes octocat-wave {
    0% {
        transform: rotate(0deg);
    }

    20% {
        transform: rotate(-25deg);
    }

    40% {
        transform: rotate(10deg);
    }

    60% {
        transform: rotate(-25deg);
    }

    80% {
        transform: rotate(10deg);
    }

    100% {
        transform: rotate(0deg);
    }
    }

    @media (max-width: 500px) {
    .github-corner:hover .octo-arm {
        animation: none;
    }

    .github-corner .octo-arm {
        animation: octocat-wave 560ms ease-in-out;
    }
    }
    """
    st.sidebar.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    st.sidebar.markdown(html_content, unsafe_allow_html=True)
    
    st.sidebar.title("Audio Visualizer")
    
    # Uploading a file format mp3/wav.
    audio_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3"])
    

    if audio_file is not None:
        audio_bytes = audio_file.read()  # Reading audio file.
        
        # Now you can use the audio bytes to process the audio data.
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        
        # Display audio length and sample rate.
        audio_length = len(y) // sr
        frames_per_second = len(y) // audio_length
        audio_format = audio_file.type.split("/")[-1].upper()
        # samples = np.arange(sr * audio_length)
        samples = np.arange(len(y))
        frequency = 440; volume = 10000; audio_data = volume * np.sin(2 * np.pi * frequency * samples / sr)[1:]

        # Information in table format.
        st.subheader("Audio Information")
        info_data = {
            "Items": ["Audio Length (Sec)", "Sample Rate (Hz)", "Frames Per Second (FPS)", "Format of Audio", "Samples", "Audio Data"],
            "Values": [audio_length, sr, frames_per_second, audio_format, samples, audio_data]
        }
        st.table(info_data)

        # Visualize uploaded audio.
        st.subheader("Uploaded Audio")
        st.audio(audio_bytes, format="mp3/wav")

        options = ["Chromagram", "MFCCs", "Waveform", "Spectrogram", "RMS Curve", "Spectrogram with Threshold", "STFT-Mel-Chroma", "Power Spectral Density (PSD)", "Cepstrum", "Autocorrelation Function",
                   "Wavelet Transform", "Tonal Analysis", "Timbre Features"]

        disabled_services = True

        if disabled_services:
            options = [opt for opt in options if opt != "Waveform"]

        # Sidebar option menu starts.
        selected_option = st.sidebar.radio("Select Visualization", options)

        if selected_option == "Chromagram":
            st.subheader("Chromagram")
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
            plt_chromagram = plot_chromagram(chromagram, sr)
            st.pyplot(plt_chromagram)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Chromagram**
                    - **About**: A Chromagram or Chroma feature represents the energy distribution of different pitch classes (e.g., C, C#, D, ...) in a piece of audio.
                    - **Why Use**: Chromagrams are useful for analyzing harmonic and melodic content of music.
                    - **When to Use**: Use Chromagrams for tasks like chord recognition, key detection, and music similarity.
                    - **Main Purpose**: To visualize pitch content over time, helping in understanding the harmonic structure of the audio.

                    ![Chromagram](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Chromagram.svg/1200px-Chromagram.svg.png)
                """)

        elif selected_option == "MFCCs":
            st.subheader("MFCCs")
            plt_mfccs = librosa.feature.mfcc(y=y, sr=sr)
            mfccs = plot_mfccs(plt_mfccs, sr)
            st.pyplot(mfccs)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **MFCCs (Mel Frequency Cepstral Coefficients)**
                    - **About**: MFCCs represent the short-term power spectrum of sound, emphasizing perceptually relevant features.
                    - **Why Use**: MFCCs are widely used in speech and audio processing because they capture important features for distinguishing sounds.
                    - **When to Use**: Use MFCCs for tasks like speech recognition, speaker identification, and music genre classification.
                    - **Main Purpose**: To extract features from audio that correlate with how humans perceive sound, making it useful for various audio analysis tasks.

                    ![MFCCs](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/MFCC.png/300px-MFCC.png)
                """)
                
        elif selected_option == "Waveform":
            st.subheader("Waveform")
            plt_waveform = plot_waveform(y, sr)
            st.pyplot(plt_waveform)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Waveform**
                    - **About**: The waveform displays the amplitude of the audio signal over time.
                    - **Why Use**: It provides a simple visual representation of the audio signal's amplitude variations.
                    - **When to Use**: Use it when you need to analyze the temporal structure of the audio signal.
                    - **Main Purpose**: To visualize the amplitude changes of the audio signal over time.

                    ![Waveform](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Audio_waveform.svg/800px-Audio_waveform.svg.png)
                """)
                
        elif selected_option == "Spectrogram":
            st.subheader("Spectrogram")
            plt_spec = plot_spectrogram(y, sr)
            st.pyplot(plt_spec)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram**
                    - **About**: A Spectrogram is a visual representation of the spectrum of frequencies in a signal as it varies with time.
                    - **Why Use**: Spectrograms help in understanding how the spectral content of a signal changes over time.
                    - **When to Use**: Use Spectrograms for tasks like speech analysis, music analysis, and identifying patterns in audio.
                    - **Main Purpose**: To visualize the frequency content of a signal over time, aiding in the analysis of audio signals.

                    ![Spectrogram](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Audio_spectrogram_example.png/1200px-Audio_spectrogram_example.png)
                """)

        elif selected_option == "RMS Curve":
            st.subheader("Root-Mean-Square (RMS) Curve and Probability RMS Curve")
            rms_curve = plot_rms_curve(y, sr)
            prob_curve = plot_probability_curve(y, sr)
            st.pyplot(rms_curve)
            st.pyplot(prob_curve)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **RMS Curve**
                    - **About**: The RMS (Root-Mean-Square) curve represents the energy of the signal over time.
                    - **Why Use**: RMS is useful for measuring the loudness and dynamics of audio.
                    - **When to Use**: Use RMS for tasks like dynamic range compression, audio level monitoring, and sound detection.
                    - **Main Purpose**: To visualize the amplitude variations in the signal over time, providing insight into the dynamics of the audio.

                    **Probability RMS Curve**
                    - **About**: The Probability RMS curve represents the normalized RMS values, indicating the relative loudness over time.
                    - **Why Use**: It helps in comparing the loudness levels of different segments of audio.
                    - **When to Use**: Use Probability RMS for tasks like audio segmentation and event detection.
                    - **Main Purpose**: To normalize the loudness levels for comparative analysis.

                    ![RMS Curve](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Sinusoidal_RMS.svg/1200px-Sinusoidal_RMS.svg.png)
                """)

        elif selected_option == "Spectrogram with Threshold":
            st.subheader("Spectrogram with Threshold")
            spec_to_threshold = plot_spectrogram_with_threshold(y, sr)
            st.pyplot(spec_to_threshold)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram with Threshold**
                    - **About**: This is a Spectrogram with an added threshold line to highlight significant frequency components.
                    - **Why Use**: Helps in identifying and isolating important frequency components in the audio signal.
                    - **When to Use**: Use this for tasks like noise reduction, signal enhancement, and identifying key features in the audio.
                    - **Main Purpose**: To filter out noise and focus on significant frequency components, making it easier to analyze important parts of the signal.

                    ![Spectrogram with Threshold](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Audio_spectrum_animation.gif/250px-Audio_spectrum_animation.gif)
                """)
            
        elif selected_option == "STFT-Mel-Chroma":
            st.subheader("Combined Visualization")
            stft = compute_stft(y, sr)
            mel_spec = compute_mel(y, sr)
            chroma = compute_chroma(y, sr)
            
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

            # Plot STFT
            ax[0].imshow(stft, aspect='auto', origin='lower', cmap='inferno', extent=[0, len(y)/sr, 0, sr/2])
            ax[0].set_title('STFT (log scale)')
            ax[0].set_ylabel('Frequency (Hz)')

            # Plot Mel spectrogram
            ax[1].imshow(mel_spec, aspect='auto', origin='lower', cmap='inferno', extent=[0, len(y)/sr, 0, sr/2])
            ax[1].set_title('Mel Spectrogram')
            ax[1].set_ylabel('Frequency (Hz)')

            # Plot Chroma
            ax[2].imshow(chroma, aspect='auto', origin='lower', cmap='inferno', extent=[0, len(y)/sr, 0, 12])
            ax[2].set_title('Chroma')
            ax[2].set_xlabel('Time (s)')
            ax[2].set_ylabel('Pitch Class')

            plt.tight_layout()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Combined Visualization (STFT, Mel Spectrogram, Chroma)**
                    - **About**: This visualization combines the Short-Time Fourier Transform (STFT), Mel Spectrogram, and Chroma features in a single view.
                    - **Why Use**: Provides a comprehensive overview of the audio's time-frequency characteristics.
                    - **When to Use**: Use this for in-depth audio analysis, where understanding both harmonic and spectral features is important.
                    - **Main Purpose**: To offer a detailed view of the audio signal, highlighting different aspects like pitch, timbre, and spectral content.

                    ![Combined Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Spectrogram-19thCentrury-AcousticCommunication.jpg/1024px-Spectrogram-19thCentrury-AcousticCommunication.jpg)
                """)
            
        elif selected_option == "Power Spectral Density (PSD)":
            st.subheader("PSD")
            plt_psd = plot_power_spectral_density(y, sr)
            st.pyplot(plt_psd)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram Animation**
                    - **About**: This animation shows the evolution of the spectrogram over time.
                    - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
                    - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
                    - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

                    [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
                """)

        elif selected_option == "Cepstrum":
            st.subheader("Cepstrum")
            plt_cepstrum = plot_cepstrum(y, sr)
            st.pyplot(plt_cepstrum)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram Animation**
                    - **About**: This animation shows the evolution of the spectrogram over time.
                    - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
                    - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
                    - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

                    [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
                """)
        
        elif selected_option == "Autocorrelation Function":
            st.subheader("Autocorrelation Function")
            plt_autocorrelation = plot_autocorrelation(y, sr)
            st.pyplot(plt_autocorrelation)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram Animation**
                    - **About**: This animation shows the evolution of the spectrogram over time.
                    - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
                    - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
                    - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

                    [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
                """)

        elif selected_option == "Wavelet Transform":
            st.subheader("Wavelet Transform")
            plt_wavlet = plot_wavelet_transform(y, sr)
            st.pyplot(plt_wavlet)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram Animation**
                    - **About**: This animation shows the evolution of the spectrogram over time.
                    - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
                    - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
                    - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

                    [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
        
                """)
        
        elif selected_option == "Tonal Analysis":
            st.subheader("Tonal Analysis")
            plt_tonal = plot_tonal_analysis(y, sr)
            st.pyplot(plt_tonal)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram Animation**
                    - **About**: This animation shows the evolution of the spectrogram over time.
                    - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
                    - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
                    - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

                    [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
                """)
        
        elif selected_option == "Timbre Features":
            st.subheader("Timbre Features Analysis")
            plt_timbre = plot_timbre_features(y, sr)
            st.pyplot(plt_timbre)
            with st.expander("Detail Explanation"):
                st.markdown("""
                    **Spectrogram Animation**
                    - **About**: This animation shows the evolution of the spectrogram over time.
                    - **Why Use**: Useful for observing changes in the spectral content of the audio signal dynamically.
                    - **When to Use**: Use this for detailed time-frequency analysis and to observe transient events in the audio.
                    - **Main Purpose**: To visualize how the frequency components of the audio signal evolve over time in an animated format.

                    [Spectrogram Animation](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Animated_spectrogram_example.gif/800px-Animated_spectrogram_example.gif)
                """)

if __name__ == "__main__":
    main()