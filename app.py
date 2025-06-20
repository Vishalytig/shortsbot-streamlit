import streamlit as st
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
import os, tempfile

st.title("🎬 ShortsBot - Auto Clip Highlights from YouTube")

keywords_input = st.text_input("Keywords (comma separated):", "important,summary,highlight")
youtube_url = st.text_input("YouTube URL here:")

if st.button("Create Shorts"):
    if not youtube_url:
        st.error("Paste a YouTube URL first!")
    else:
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
        model = WhisperModel("small", device="cpu", compute_type="int8")

        with st.spinner("📥 Downloading video..."):
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(file_extension="mp4", progressive=True).order_by('resolution').desc().first()
            temp_video = os.path.join(tempfile.gettempdir(), "video.mp4")
            stream.download(filename=temp_video)

        with st.spinner("🧠 Transcribing video..."):
            segments, _ = model.transcribe(temp_video, beam_size=5)

        highlights = [
            (s.start, s.end, s.text)
            for s in segments
            if any(kw in s.text.lower() for kw in keywords)
        ]

        if not highlights:
            st.warning("No highlights found with those keywords.")
        else:
            os.makedirs("clips", exist_ok=True)
            base = VideoFileClip(temp_video)

            st.success(f"✅ Created {len(highlights)} clip(s):")
            for i, (start, end, text) in enumerate(highlights):
                outfile = f"clips/clip_{i+1}.mp4"
                base.subclip(start, end).write_videofile(outfile, codec="libx264", audio_codec="aac", verbose=False)
                st.video(outfile)
                with open(outfile, "rb") as fp:
                    st.download_button(f"⬇️ Download clip_{i+1}.mp4", fp, file_name=f"clip_{i+1}.mp4")
            base.close()
streamlit
pytube
moviepy
faster-whisper
