import streamlit as st
import os
import tempfile
import subprocess
import ffmpeg
from faster_whisper import WhisperModel
from openai import OpenAI
import re

# --- Config ---
MODEL_SIZE = "tiny"
MIN_CLIP_LENGTH = 25
MAX_CLIP_LENGTH = 60
MAX_SEGMENTS = 7

# --- UI ---
st.set_page_config(page_title="🔥 ShortsBot", layout="centered")
st.title("🔥 ShortsBot: AI-Powered Viral Clip Generator")
st.markdown("Paste a YouTube video (public). Let me find standout 25–60 sec highlights!")

youtube_url = st.text_input("📻 Paste YouTube link")
keywords_input = st.text_input("🔑 Target keywords (optional, comma-separated)", "summary,important,key point")
use_gpt = st.checkbox("🤖 Use GPT to smartly pick highlights (slower)")

def download_youtube_video(url, output_path):
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "-o", output_path,
        "--no-playlist",
        "--user-agent", "Mozilla/5.0"
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Download failed: {e}")

def cut_clip_ffmpeg(input_path, start, end, output_path):
    (
        ffmpeg
        .input(input_path, ss=start, to=end)
        .output(output_path, codec='libx264', acodec='aac', loglevel='error')
        .run(overwrite_output=True)
    )

if st.button("🎨 Generate Shorts"):
    if not youtube_url:
        st.warning("Please paste a YouTube link first!")
        st.stop()

    temp_path = os.path.join(tempfile.gettempdir(), "shortsbot_video.mp4")
    try:
        download_youtube_video(youtube_url, temp_path)
    except Exception as e:
        st.error(f"Failed to download the video: {e}")
        st.stop()

    st.info("🔎 Transcribing audio—this may take a moment.")
    whisper = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(temp_path, beam_size=5)

    highlights = []
    if use_gpt:
        st.info("🧠 Using GPT to analyze transcript and select highlights")
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        full_text = "\n".join([seg.text for seg in segments])
        prompt = ("You are a video editor. Identify 3–7 highlight clips (25–60s) "
                  "from this transcript that are emotional, funny, or impactful. "
                  "Give start/end in mm:ss format like '01:23 - 01:45: explanation.'\n\n"
                  f"Transcript:\n{full_text}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        gpt_text = response.choices[0].message.content
        times = re.findall(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*:?:?\s*(.+)", gpt_text)
        def to_sec(ts):
            mm, ss = map(int, ts.split(":"))
            return mm * 60 + ss
        for start_ts, end_ts, _ in times:
            s, e = to_sec(start_ts), to_sec(end_ts)
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
        st.warning("No suitable highlights found. Try different keywords or GPT option.")
        st.stop()

    st.success(f"Creating {len(highlights)} short clip(s)...")
    os.makedirs("viral_clips", exist_ok=True)

    for i, (start, end) in enumerate(highlights, start=1):
        out_path = f"viral_clips/short_{i}.mp4"
        cut_clip_ffmpeg(temp_path, start, end, out_path)
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(f"⬇️ Download Short {i}", f, file_name=os.path.basename(out_path))

    st.balloons()
    st.success("✅ All set! Ready for Shorts, Reels, TikTok!")
