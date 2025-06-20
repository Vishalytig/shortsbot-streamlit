import streamlit as st
import os
import tempfile
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from faster_whisper import WhisperModel
from openai import OpenAI

# --- CONFIG ---
MODEL_SIZE = "tiny"
MIN_CLIP_LENGTH = 25
MAX_CLIP_LENGTH = 60
MAX_SEGMENTS = 7
BG_MUSIC = "background_music.mp3"  # Ensure this file is in the project root

# --- UI ---
st.set_page_config(page_title="🔥 ShortsBot", layout="centered")
st.title("🔥 ShortsBot: AI-Powered Viral Clip Generator")
st.markdown("Paste a YouTube video. I’ll find the best moments and turn them into shorts — with music and AI brains!")

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
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        full_text = "\n".join([seg.text for seg in segments])
        prompt = f"You are a smart video editor. Find 3-7 short highlights (25-60 seconds long) from this transcript that are funny, emotional, or impactful. Give exact start and end times with 1 sentence summary.\nTranscript:\n{full_text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        import re
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

    st.success(f"Creating {len(highlights)} clips with background music...")
    os.makedirs("viral_clips", exist_ok=True)
    main_video = VideoFileClip(temp_path)
    bg_music = AudioFileClip(BG_MUSIC).volumex(0.2) if os.path.exists(BG_MUSIC) else None

    for i, (start, end) in enumerate(highlights):
        clip = main_video.subclip(start, end)
        if bg_music:
            bg_audio = bg_music.subclip(0, clip.duration).set_duration(clip.duration)
            final = clip.set_audio(clip.audio.volumex(0.8).fx(lambda a: a.set_duration(clip.duration)).fx(lambda a: a.set_start(0)).fx(lambda a: a.set_end(clip.duration)))
            final = final.set_audio(bg_audio)
        else:
            final = clip
        out_path = f"viral_clips/short_{i+1}.mp4"
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False)
        st.video(out_path)
        with open(out_path, "rb") as f:
            st.download_button(f"⬇️ Download Short {i+1}", f, file_name=os.path.basename(out_path))

    main_video.close()
    st.balloons()
    st.success("Done! Upload these to Shorts, Reels or TikTok 🎉")
