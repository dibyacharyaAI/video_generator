
import os
import logging
from flask import Flask, request, jsonify, send_from_directory

from .agents.prompt_agent import PromptRefinementAgent
from .services.rag_service import RAGService
from .agents.director_agent import DirectorAgent
from .agents.script_agent import ScriptAgent
from .agents.video_agent import VideoAgent
from .agents.review_agent import ReviewAgent

# Initialize Flask app
app = Flask(__name__, static_folder=None)
logging.basicConfig(level=logging.DEBUG)

# Initialize agents and services
prompt_agent = PromptRefinementAgent()
rag_service = RAGService()
director_agent = DirectorAgent(rag_service)
script_agent = ScriptAgent()
video_agent = VideoAgent()
review_agent = ReviewAgent()

# Directory for generated videos
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "generated_videos")

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend UI"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    frontend_path = os.path.join(base_dir, "frontend")
    return send_from_directory(frontend_path, "index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    """Run the full pipeline on the input prompt"""
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data["prompt"]
    try:
        # Step 0: refine prompt
        refined_prompt = prompt_agent.refine(prompt)
        logging.debug(f"Refined prompt: {refined_prompt}")

        # Step 1: retrieve context using RAG
        context = rag_service.get_medical_context(refined_prompt)
        logging.debug(f"Context retrieved: {context}")

        # Step 2: plan storyboard using refined prompt and context
        scenes = director_agent.plan(refined_prompt, context)
        logging.debug(f"Storyboard generated: {[scene.dict() for scene in scenes]}")

        # Step 3: generate narrations (Script Agent)
        scenes = script_agent.generate_scripts(scenes)
        logging.debug(f"Narrations added: {[scene.dict() for scene in scenes]}")

        # Combine narrations into transcript
        transcript = " ".join(scene.narration for scene in scenes)
        logging.debug(f"Transcript: {transcript}")

        # Step 4: generate video clips
        clips = video_agent.generate_videos(scenes)
        logging.debug(f"Video clips generated: {[clip.dict() for clip in clips]}")

        # Step 5: review quality
        review = review_agent.review(scenes)
        logging.debug(f"Review: {review}")

        # Return JSON with all fields
        return jsonify({
            "refined_prompt": refined_prompt,
            "context": context,
            "storyboard": [scene.dict() for scene in scenes],
            "transcript": transcript,
            "clips": [clip.dict() for clip in clips],
            "review": review
        })
    except Exception as e:
        logging.exception("Error during generation")
        return jsonify({"error": str(e)}), 500

@app.route("/videos/<path:filename>", methods=["GET"])
def serve_generated_video(filename: str):
    """Serve generated video files"""
    return send_from_directory(VIDEO_DIR, filename)

@app.route("/<path:path>", methods=["GET"])
def static_proxy(path: str):
    """Serve other static files from the frontend folder"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    frontend_path = os.path.join(base_dir, "frontend")
    return send_from_directory(frontend_path, path)

def run():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    run()
