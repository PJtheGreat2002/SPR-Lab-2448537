import streamlit as st
import numpy as np
import pandas as pd
import os
import vosk
import wave
import speech_recognition as sr
import sounddevice as sd
import wavio

from scipy.io import wavfile

SAMPLE_RATE = 16000
DURATION = 5
MODEL_PATH = "vosk-model-small-en-us-0.15"

# Audio types for comparison
audio_types = [
    "Clear male voice",
    "Clear female voice",
    "Fast speech",
    "Noisy background",
    "Soft voice"
]

# Initial DataFrame for table
if "comparison_df" not in st.session_state:
    st.session_state.comparison_df = pd.DataFrame({
        "Audio Type": audio_types,
        "Whisper Output": [""]*5,
        "Vosk Output": [""]*5,
        "Google API Output": [""]*5,
        "Any other python libraries can be added ..": [""]*5,
        "Notes on Accuracy": [""]*5
    })

st.title("Speech-to-Text Comparative Analysis")

# --- 1. Select audio type for this test ---
st.header("1. Select Audio Scenario")
audio_type_selected = st.selectbox("Which audio type are you testing right now?", audio_types)

row_idx = audio_types.index(audio_type_selected)

# --- 2. Audio Input (Mic or File) ---
st.header("2. Input Audio")
input_method = st.radio("Audio input method:", ("Record from microphone", "Upload .wav or .flac file"))

audio_file = None
if input_method == "Record from microphone":
    if st.button("Record"):
        st.info("Speak something...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wavio.write("audio_input.wav", recording, SAMPLE_RATE, sampwidth=2)
        audio_file = "audio_input.wav"
        st.audio(audio_file, format="audio/wav")
elif input_method == "Upload .wav or .flac file":
    uploaded_file = st.file_uploader("Upload file", type=["wav", "flac"])
    if uploaded_file is not None:
        audio_file = "uploaded_audio.wav"
        with open(audio_file, "wb") as f:
            f.write(uploaded_file.read())
        st.success("File uploaded")
        st.audio(audio_file, format="audio/wav")

# --- 3. Visualize waveform/spectrogram ---
if audio_file:
    st.header("3. Audio Visualization")
    try:
        rate, data = wavfile.read(audio_file)
        if len(data.shape) > 1:
            data = data[:,0]
        t = np.linspace(0, len(data)/rate, num=len(data))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,2))
        ax.plot(t, data)
        ax.set_title("Waveform")
        st.pyplot(fig)
        fig2, ax2 = plt.subplots(figsize=(10,3))
        Pxx, freqs, bins, im = ax2.specgram(data, Fs=rate, NFFT=1024, noverlap=512, cmap='plasma')
        ax2.set_title("Spectrogram")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Cannot plot: {e}")

# --- 4. Recognition Functions ---

def recognize_whisper(audio_file):
    # Dummy placeholder; replace with real Whisper inference if available
    return "Dummy Whisper result for demo"  # replace with real code as needed

def recognize_vosk(audio_file):
    try:
        model = vosk.Model(MODEL_PATH)
        wf = wave.open(audio_file, "rb")
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(rec.Result())
        results.append(rec.FinalResult())
        import json
        text = ""
        for res in results:
            text += json.loads(res).get('text', '') + " "
        return text.strip()
    except Exception as e:
        return f"Vosk Error: {e}"

def recognize_google(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand audio. Please try speaking more clearly."
    except sr.RequestError:
        return "Google Speech Recognition service unavailable."
    except Exception as e:
        return f"Google API Error: {e}"

# --- 5. Run Recognition and Auto-fill table ---
if audio_file:
    run_recognition = st.button("Run Speech Recognition and Fill Table Row")
    if run_recognition:
        with st.spinner("Recognizing (Whisper)..."):
            whisper_out = recognize_whisper(audio_file)
        with st.spinner("Recognizing (Vosk)..."):
            vosk_out = recognize_vosk(audio_file)
        with st.spinner("Recognizing (Google)..."):
            google_out = recognize_google(audio_file)
        
        # Update session state table automatically
        st.session_state.comparison_df.loc[row_idx, "Whisper Output"] = whisper_out
        st.session_state.comparison_df.loc[row_idx, "Vosk Output"] = vosk_out
        st.session_state.comparison_df.loc[row_idx, "Google API Output"] = google_out
        st.success("Table updated! You can edit Notes below, then export.")

# --- 6. Editable Table and Export ---
st.header("4. Comparative Table (Editable and Exportable)")
edited_df = st.data_editor(
    st.session_state.comparison_df,
    column_config={"Audio Type": st.column_config.Column(disabled=True)},
    width='stretch',
    key="comparison_table"
)
st.session_state.comparison_df = edited_df  # Persist changes

st.download_button(
    label="Download as CSV",
    data=edited_df.to_csv(index=False),
    file_name="comparative_analysis_results.csv",
    mime="text/csv"
)

st.info("Table auto-fills the selected row after you run recognition. You can further edit notes and download the CSV.")

