import os
import traceback
from pathlib import Path
from datetime import datetime

import streamlit as st

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

# --- Page config ---
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

# --- Debug panel (helps when boss sees blank / no video) ---
with st.expander("🛠 Debug / Runtime Info", expanded=False):
    st.write("**Environment (best-effort):**")
    st.code(
        "\n".join(
            [
                f"DIFFUSION_API_URL: {os.getenv('DIFFUSION_API_URL')}",
                f"Has DIFFUSION_API_KEY: {bool(os.getenv('DIFFUSION_API_KEY'))}",
                f"Has HF_TOKEN: {bool(os.getenv('HF_TOKEN'))}",
                f"VIDEO_FPS: {os.getenv('VIDEO_FPS')}",
                f"VIDEO_CLIP_SECONDS: {os.getenv('VIDEO_CLIP_SECONDS')}",
            ]
        )
    )
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        st.write("**Streamlit Secrets available keys:**")
        st.code(", ".join(sorted(list(st.secrets.keys()))))
    else:
        st.write("**Streamlit Secrets:** not available or empty.")
    st.write("If diffusion fails intermittently, check Cloud logs for **401/429/503/timeouts**.")

# If imports failed, show error instead of blank screen
if not PIPELINE_OK:
    st.error("❌ Backend pipeline import failed. Check the traceback below.")
    st.code(ERROR_TEXT, language="python")
    st.stop()

# --- Init session state to avoid blank after reruns ---
if "result" not in st.session_state:
    st.session_state["result"] = None

# --- Initialise agents ---
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
    clear_btn = st.button("🧹 Clear Last Result")

if clear_btn:
    st.session_state["result"] = None
    st.success("Cleared ✅")

# --- Generation ---
if generate_btn and prompt.strip():
    with st.spinner("Planning scenes, calling diffusion, and building videos..."):
        try:
            refined_prompt = prompt_agent.refine(prompt)
            context = rag_service.get_medical_context(refined_prompt)
            scenes = director_agent.plan(refined_prompt, context)
            scenes = script_agent.generate_scripts(scenes)
            transcript = " ".join(scene.narration or "" for scene in scenes)
            clips = video_agent.generate_videos(scenes)
            review = review_agent.review(scenes)

            st.session_state["result"] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input_prompt": prompt,
                "refined_prompt": refined_prompt,
                "context": context,
                "scenes": scenes,
                "clips": clips,
                "review": review,
                "transcript": transcript,
            }

        except Exception:
            st.error("❌ Error while running the generation pipeline.")
            st.code(traceback.format_exc(), language="python")
            st.stop()

    st.success("Generation complete ✅")

# --- Display (persisted) ---
result = st.session_state["result"]

if result:
    refined_prompt = result["refined_prompt"]
    context = result["context"]
    scenes = result["scenes"]
    clips = result["clips"]
    review = result["review"]
    transcript = result["transcript"]

    # LEFT: text info
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
        st.caption(f"Generated at: {result.get('timestamp')}")

    # RIGHT: video clips
    with col_right:
        st.subheader("Generated Scene Clips")

        if not clips:
            st.info("No clips generated.")
        else:
            for clip in clips:
                st.markdown(f"**{clip.scene_title}**")

                # clip.video_url can be:
                #  - "/videos/scene_x.mp4"
                #  - "demo_pool/xxx.mp4"
                #  - "/videos/demo_pool/xxx.mp4"
                rel = str(clip.video_url).lstrip("/")        # remove leading slash
                if rel.startswith("videos/"):
                    rel = rel[len("videos/"):]              # map to generated_videos root

                video_path = VIDEO_DIR / rel

                # Debug info per video
                with st.expander(f"Path details: {clip.scene_title}", expanded=False):
                    st.code(f"clip.video_url: {clip.video_url}\nresolved: {video_path}")

                if video_path.exists():
                    st.video(str(video_path))
                else:
                    st.warning(f"Video file not found: {video_path}\nThis can happen if generation failed or path mapping differs.")
else:
    with col_right:
        st.info("Enter a procedure description and click **Generate** to run the pipeline.")
