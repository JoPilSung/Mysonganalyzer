import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="자작곡 분석기", layout="centered")
st.title("자작곡 분석기")
st.markdown("MP3/WAV 업로드 시 코드와 멜로디를 분석합니다.")

uploaded_file = st.file_uploader("음원 파일 업로드", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.spinner("분석 중..."):
        y, sr = librosa.load(uploaded_file, sr=None)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chord_index = np.argmax(np.mean(chroma, axis=1))
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = pitch_classes[chord_index]

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.5)
        ax.set_title(f"Estimated Key: {key}")
        st.pyplot(fig)

        st.success(f"예상 코드(Key): {key}")
