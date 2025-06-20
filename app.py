import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
import os
import tempfile
import subprocess

# Helper to download video using yt-dlp
def download_youtube_video(url, output_path):
    ydl_cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", output_path,
        url
    ]
    subprocess.run(ydl_cmd, check=True)

# Streamlit UI
st.set_page_config(page_title="ShortsBot", layout="centered")
st.title("🎬 ShortsBot - Turn Long YouTube Videos into Short Clips")
st.markdown("Give me a long video and some keywords, and I’ll find the magic moments!")

keywords_input = st.text_input("🎯 Keywords (comma-separated)", "important,summary,highlight")
youtube_url = st.text_input("📺 Paste a YouTube URL")

if st.button("✨ Create Shorts"):
    if not youtube_url:
        st.warning("Please paste a YouTube link first.")
    else:
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
        model = WhisperModel("small", device="cpu", compute_type="int8")

        with st.spinner("📥 Downloading YouTube video..."):
            temp_path = os.path.join(tempfile.gettempdir(), "video.mp4")
            try:
                download_youtube_video(youtube_url, temp_path)
            except subprocess.CalledProcessError:
                st.error("🚫 Failed to download the YouTube video. Check if it's public and available.")
                st.stop()

        with st.spinner("🧠 Transcribing audio..."):
            segments, _ = model.transcribe(temp_path, beam_size=5)

        highlights = [
            (seg.start, seg.end, seg.text)
            for seg in segments
            if any(kw in seg.text.lower() for kw in keywords)
        ]

        if not highlights:
            st.warning("😕 No highlights found. Try different keywords.")
        else:
            os.makedirs("clips", exist_ok=True)
            video = VideoFileClip(temp_path)
            st.success(f"✅ Found {len(highlights)} highlight(s). Here they are:")

            for i, (start, end, text) in enumerate(highlights):
                output_file = f"clips/highlight_{i+1}.mp4"
                video.subclip(start, end).write_videofile(output_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                st.video(output_file)
                with open(output_file, "rb") as f:
                    st.download_button(f"⬇️ Download highlight_{i+1}.mp4", f, file_name=f"highlight_{i+1}.mp4")
            video.close()

