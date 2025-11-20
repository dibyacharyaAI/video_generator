
# Medical Video Generation Platform (Flask + Multi-Agent)

This project implements a simplified multi-agent architecture for generating pedagogical medical videos from natural language prompts.

## Overview

The system follows the architecture described in the provided System Architecture Overview:

1. **User Interface**: A web dashboard (frontend/index.html) where surgeons or educators enter a free-text prompt describing a medical procedure and can view the results.
2. **Prompt Refinement Agent**: Takes the raw prompt and enriches it with context (e.g. adding "step-by-step tutorial" and specifying the surgeon perspective) before passing it to subsequent agents.
3. **Retrieval-Augmented Generation (RAG) Layer**: The `RAGService` retrieves factual medical context for the refined prompt from a local JSON of medical facts. In a production system, this would interface with a vector database (e.g. Milvus) and LlamaIndex.
4. **Director (Planning) Agent**: Breaks down the refined prompt and retrieved context into a scene-by-scene storyboard. Each scene includes an ID, title, description, and any relevant context.
5. **Script (Content) Agent**: Generates narration text for each scene based on its description.
6. **Video Generation Agent**: For each scene, the `VideoAgent` uses `VanClient` to generate a video clip. If a text-to-image diffusion API is configured (via environment variables), it will call the API to create an image and animate it with OpenCV. Otherwise, it falls back to a locally generated animated card with the scene text.
7. **Review Agent**: Performs a basic automated quality check on the scenes and narrations, returning a score and comments.
8. **API Response**: Combines all the above into a single JSON response containing the refined prompt, retrieved context, storyboard, transcript (concatenated narrations), generated clip URLs, and review.

## Running Locally

1. Install Python 3.8+ and create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Optionally set environment variables for diffusion API:

   - `DIFFUSION_API_URL` – Endpoint of a text-to-image diffusion service.
   - `DIFFUSION_API_KEY` – API key if required by the service.
   - `DIFFUSION_API_MODE` – `"bytes"` if the API returns image bytes; `"url"` if it returns a JSON object with an `image_url`.
   - Additional variables like `VIDEO_CLIP_SECONDS`, `VIDEO_WIDTH`, `VIDEO_HEIGHT`, `VIDEO_FPS` can adjust clip settings.

   If these variables are not set, the system uses a built-in OpenCV fallback to generate an animated card for each scene.

3. Run the Flask server:

   ```bash
   python -m backend.app
   ```

   The server will start on port 5000.

4. Open your browser and navigate to `http://localhost:5000/`. Enter a medical procedure description (e.g. "Laparoscopic cholecystectomy tutorial for surgery residents") and click **Generate Video Plan & Clips**. The application returns a JSON structure with:

   - `refined_prompt`: the enriched prompt passed to the pipeline.
   - `context`: retrieved medical context used to ground the scenes.
   - `storyboard`: a list of scene objects (ID, title, description, narration).
   - `transcript`: concatenated narration text for all scenes.
   - `clips`: a list of video clip objects with `id`, `scene_title`, and `video_url`.
   - `review`: a basic QA score and comments.

   Below the JSON, each clip URL is rendered as a `<video>` tag, so you can preview the generated video segments.

## Project Structure

- `backend/app.py`: Flask entry point defining API routes and orchestrating the multi-agent pipeline.
- `backend/agents/`: Contains agent classes:
  - `prompt_agent.py`: Enriches user prompts.
  - `director_agent.py`: Produces the storyboard.
  - `script_agent.py`: Generates narration for each scene.
  - `video_agent.py`: Generates videos per scene via `VanClient`.
  - `review_agent.py`: Performs a simple quality check.
- `backend/services/`: Utility services:
  - `rag_service.py`: Loads a static JSON of medical facts and retrieves relevant context.
  - `van_client.py`: Handles diffusion API calls or OpenCV fallback for video creation.
- `backend/generated_videos/`: Directory where generated video clips are saved.
- `frontend/index.html`: Minimal web interface to interact with the API.
- `requirements.txt`: Python dependencies.
- `README.md`: This documentation.

## Notes

- This project is a simplified prototype for demonstration purposes. In a production environment, the RAG service would connect to a vector database (e.g. Milvus) and use an actual LLM or search engine to retrieve authoritative medical content. The director and script agents would employ large language models to plan and generate content. The video generation agent would call advanced video models like Veo or Sora, and the review agent would leverage vision-language models (VLMs) for safety checks.

- The current implementation uses OpenCV to generate simple animated videos if no diffusion API is configured. These animations display the scene title and description with a progress bar. To enable realistic video generation, configure `DIFFUSION_API_URL` and `DIFFUSION_API_KEY` to point to a text-to-image diffusion service that can generate a relevant image for each scene.
