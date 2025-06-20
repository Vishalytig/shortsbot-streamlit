import streamlit as st
import os
import tempfile
import subprocess
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import openai
import re

# --- CONFIG ---
MODEL_SIZE = "tiny"
MIN_CLIP_LENGTH = 25
MAX_CLIP_LENGTH = 60
MAX_SEGMENTS = 7

# --- UI ---
st.set_page_config(page_title="🔥 ShortsBot", layout="centered")
st.title("🔥 ShortsBot: AI-Powered Viral Clip Generator")
st.markdown("Paste a YouTube video. I’ll find the best moments and turn them into shorts!")

youtube_url = st.text_input("📺 Paste YouTube link")
keywords_input = st.text_input("🔑 Target keywords (optional, comma-separated)", "summary,important,key point")
use_gpt = st.checkbox("🤖 Use GPT to auto-pick best clips (slower, smarter)")

def download_youtube_video(url, output_path):
    cmd = ["yt-dlp", "-f", "best[ext=mp4]", "-o", output_path, url]
    subprocess.run(cmd, check=True)

if st.button("🎬 Generate Shorts"):
    if not youtube_url:
        st.warning("Paste a YouTube link!")
        st.stop()

    temp_path = os.path.join(tempfile.gettempdir(), "shortsbot_video.mp4")
    try:
        download_youtube_video(youtube_url, temp_path)
    except Exception:
        st.error("Failed to download video.")
        st.stop()

    st.info("🔎 Transcribing... this may take a minute")
    whisper = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(temp_path, beam_size=5)

    highlights = []
    if use_gpt:
        st.info("🧠 Asking GPT to find highlights...")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        full_text = "\n".join([seg.text for seg in segments])
        prompt = f"You are a smart video editor. Find 3-7 short highlights (25-60 seconds long) from this transcript that are funny, emotional, or impactful. Give exact start and end times with 1 sentence summary.\nTranscript:\n{full_text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        gpt_text = response.choices[0].message.content
        times = re.findall(r"(\d+:\d+).*?\-.*?(\d+:\d+).*?: (.+)", gpt_text)
        def to_sec(t): return sum(int(x) * 60**i for i, x in enumerate(reversed(t.split(":"))))
        for start, end, desc in times:
            s, e = to_sec(start), to_sec(end)
            if MIN_CLIP_LENGTH <= e - s <= MAX_CLIP_LENGTH:
                highlights.append((s, e))
    else:
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
        for seg in segments:
            dur = seg.end - seg.start
            if any(k in seg.text.lower() for k in keywords) and MIN_CLIP_LENGTH <= dur <= MAX_CLIP_LENGTH:
                highlights.append((seg.start, seg.end))
            if len(highlights) >= MAX_SEGMENTS:
                break

    if not highlights:
        st.warning("No suitable clips found.")
        st.stop()

    st.success(f"Creating {len(highlights)} clips...")
    os.makedirs("viral_clips", exist_ok=True)
    main_video = VideoFileClip(temp_path)

    for i, (start, end) in enumerate(highlights):
        clip = main_video.subclip(start, end)
        out_path = f"viral_clips/short_{i+1}.mp4"
        clip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False)
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(f"⬇️ Download Short {i+1}", f, file_name=os.path.basename(out_path))

    main_video.close()
    st.balloons()
    st.success("Done! Upload these to Shorts, Reels or TikTok 🎉")
