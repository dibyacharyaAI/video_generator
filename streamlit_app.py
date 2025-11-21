import os
import uuid
import random
from typing import Optional, List

import cv2
import numpy as np
import requests

# Streamlit optional import (Cloud ke liye)
try:
    import streamlit as st
except ImportError:
    st = None


# --- Safely try importing backend pipeline ---
PIPELINE_OK = True
ERROR_TEXT = ""

try:
    from backend.agents.prompt_agent import PromptRefinementAgent
    from backend.services.rag_service import RAGService
    from backend.agents.director_agent import DirectorAgent
    from backend.agents.script_agent import ScriptAgent
    from backend.agents.video_agent import VideoAgent
    from backend.agents.review_agent import ReviewAgent
except Exception:
    PIPELINE_OK = False
    ERROR_TEXT = traceback.format_exc()

# --- Basic Streamlit page layout ---
st.set_page_config(page_title="Medical Video Generator", layout="wide")
st.title("🧠⚕️ AI-based Medical Video Generator (Demo)")

st.markdown(
    """
This demo shows how a **single prompt** becomes a medical training video:

1. Prompt is refined (surgeon-perspective, tutorial framing)  
2. Medical context is retrieved (RAG stub)  
3. A scene-by-scene storyboard is planned  
4. Narration/script is generated  
5. **Stable Diffusion XL** generates images for each scene  
6. Images are converted to animated **MP4 clips** with OpenCV  
"""
)

# If imports failed, show error instead of blank screen
if not PIPELINE_OK:
    st.error("❌ Backend pipeline import failed. Check the traceback below.")
    st.code(ERROR_TEXT, language="python")
    st.stop()

# --- Initialise agents if imports succeed ---
prompt_agent = PromptRefinementAgent()
rag_service = RAGService()
director_agent = DirectorAgent(rag_service)
script_agent = ScriptAgent()
video_agent = VideoAgent()
review_agent = ReviewAgent()

# Directory where VideoAgent writes videos
VIDEO_DIR = Path(__file__).parent / "backend" / "generated_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

default_prompt = "Create a step-by-step laparoscopic cholecystectomy tutorial for surgery residents."
prompt = st.text_area("Enter procedure description:", value=default_prompt, height=140)

col_left, col_right = st.columns([1, 1])

with col_left:
    generate_btn = st.button("🚀 Generate Video Plan & Clips", type="primary")

if generate_btn and prompt.strip():
    with st.spinner("Thinking, planning scenes, calling diffusion, and building videos..."):
        try:
            # 1. Prompt refinement
            refined_prompt = prompt_agent.refine(prompt)

            # 2. RAG context
            context = rag_service.get_medical_context(refined_prompt)

            # 3. Storyboard
            scenes = director_agent.plan(refined_prompt, context)

            # 4. Script / narration
            scenes = script_agent.generate_scripts(scenes)

            # Transcript (all narrations combined)
            transcript = " ".join(scene.narration or "" for scene in scenes)

            # 5. Video clips
            clips = video_agent.generate_videos(scenes)

            # 6. Review
            review = review_agent.review(scenes)

        except Exception:
            st.error("❌ Error while running the generation pipeline.")
            st.code(traceback.format_exc(), language="python")
            st.stop()

    st.success("Generation complete ✅")

    # --- LEFT: text info ---
    with col_left:
        st.subheader("Refined Prompt")
        st.code(refined_prompt, language="text")

        st.subheader("Retrieved Context")
        if context:
            for c in context:
                st.markdown(f"- {c}")
        else:
            st.write("No context retrieved.")

        st.subheader("Storyboard")
        for scene in scenes:
            st.markdown(
                f"**Scene {scene.id}: {scene.title}**  \n"
                f"*Description*: {scene.description}  \n"
                f"*Narration*: {scene.narration}"
            )

        st.subheader("Transcript")
        st.text_area("Full transcript:", value=transcript, height=150)

        st.subheader("Review")
        st.markdown(f"**Score**: {review.get('score')}  \n**Comments**: {review.get('comments')}")

    # --- RIGHT: video clips ---
    with col_right:
        st.subheader("Generated Scene Clips")

        if not clips:
            st.info("No clips generated.")
        else:
            for clip in clips:
                st.markdown(f"**{clip.scene_title}**")
                # clip.video_url is like "/videos/scene_1_xxx.mp4" → we just use the filename
                video_path = VIDEO_DIR / Path(clip.video_url).name
                if video_path.exists():
                    st.video(str(video_path))
                else:
                    st.warning(f"Video file not found: {video_path}")
else:
    with col_right:
        st.info("Enter a procedure description and click **Generate** to see the pipeline in action.")

