import os
import uuid
import time
from typing import Optional

import cv2
import numpy as np
import requests

# Streamlit optional import (Cloud secrets ke liye)
try:
    import streamlit as st
except Exception:
    st = None


class VanClient:
    """
    Video generation client using Stable Diffusion (or any text-to-image API)
    + OpenCV animation, with sensible fallbacks.

    Modes (in priority order):

    1. Demo pool mode (optional, no API needed):
       - If you put any .mp4 files into backend/generated_videos/demo_pool/,
         the client can reuse them as pre-rendered clips.

    2. Diffusion mode (recommended):
       - Calls DIFFUSION_API_URL (HF Inference router SDXL supported).
       - Uses DIFFUSION_API_KEY (or HF_TOKEN) as Bearer token.
       - Expects raw image bytes.

    3. Pure OpenCV text demo (fallback):
       - If diffusion fails or not configured.
    """

    def __init__(self, output_dir: Optional[str] = None):
        # Base directories
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if output_dir is None:
            self.video_dir = os.path.join(backend_root, "generated_videos")
        else:
            self.video_dir = output_dir

        self.image_dir = os.path.join(backend_root, "generated_images")
        self.demo_pool_dir = os.path.join(self.video_dir, "demo_pool")

        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.demo_pool_dir, exist_ok=True)

        # -----------------------------
        # Diffusion config: ENV first
        # -----------------------------
        diffusion_url = os.getenv("DIFFUSION_API_URL")
        diffusion_key = os.getenv("DIFFUSION_API_KEY") or os.getenv("HF_TOKEN")

        diffusion_mode = os.getenv("DIFFUSION_API_MODE", "bytes").lower()  # "bytes" or "url"
        negative_prompt = os.getenv("DIFFUSION_NEGATIVE_PROMPT", "")

        # -----------------------------
        # If ENV missing, try Streamlit Secrets (Cloud)
        # -----------------------------
        if (not diffusion_url or not diffusion_key) and st is not None:
            try:
                if not diffusion_url and "DIFFUSION_API_URL" in st.secrets:
                    diffusion_url = st.secrets["DIFFUSION_API_URL"]
                if not diffusion_key and "DIFFUSION_API_KEY" in st.secrets:
                    diffusion_key = st.secrets["DIFFUSION_API_KEY"]
                if (not negative_prompt) and "DIFFUSION_NEGATIVE_PROMPT" in st.secrets:
                    negative_prompt = st.secrets["DIFFUSION_NEGATIVE_PROMPT"]
                if "DIFFUSION_API_MODE" in st.secrets:
                    diffusion_mode = str(st.secrets["DIFFUSION_API_MODE"]).lower()
            except Exception as e:
                print(f"[VanClient] Could not read Streamlit secrets: {e}")

        self.diffusion_url = diffusion_url
        self.diffusion_key = diffusion_key
        self.diffusion_mode = diffusion_mode
        self.negative_prompt = negative_prompt

        # Video settings (ENV/Secrets generally both become ENV; but keep ENV for simplicity)
        self.width = int(os.getenv("VIDEO_WIDTH", "960"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "540"))
        self.fps = int(os.getenv("VIDEO_FPS", "24"))
        self.duration = int(os.getenv("VIDEO_CLIP_SECONDS", "6"))

        print(
            "[VanClient] init:",
            "diffusion_url =", self.diffusion_url,
            "| has_key =", bool(self.diffusion_key),
            "| fps =", self.fps,
            "| seconds =", self.duration,
        )

    # ------------------------------------------------------------------
    # Public main method
    # ------------------------------------------------------------------
    def generate_clip(self, scene) -> str:
        """Generate or reuse a video clip for a given scene and return filename (or relative demo_pool path)."""

        # 1) Demo pool clips first (great for guaranteed impressive demo)
        demo_choice = self._maybe_pick_demo_clip(scene)
        if demo_choice:
            return demo_choice  # e.g. "demo_pool/xyz.mp4"

        # 2) Try diffusion if configured
        if self.diffusion_url and self.diffusion_key:
            try:
                img_path = self._generate_image_via_diffusion(scene)
                return self._image_to_video(img_path, scene)
            except Exception as e:
                print(f"[VanClient] Diffusion failed: {e}. Falling back to OpenCV-only demo.")

        elif self.diffusion_url and not self.diffusion_key:
            print("[VanClient] diffusion_url is set but token missing. Falling back to OpenCV-only demo.")

        # 3) Fallback: pure OpenCV text demo
        return self._generate_text_demo_clip(scene)

    # ------------------------------------------------------------------
    # Demo pool mode
    # ------------------------------------------------------------------
    def _maybe_pick_demo_clip(self, scene) -> Optional[str]:
        """If demo_pool has any mp4 files, pick one for this scene."""
        try:
            files = [f for f in os.listdir(self.demo_pool_dir) if f.lower().endswith(".mp4")]
        except FileNotFoundError:
            return None

        if not files:
            return None

        files.sort()
        idx = (getattr(scene, "id", 1) - 1) % len(files)
        demo_file = files[idx]
        # Return relative path inside generated_videos
        return os.path.join("demo_pool", demo_file)

    # ------------------------------------------------------------------
    # Diffusion (text-to-image) mode
    # ------------------------------------------------------------------
    def _build_prompt(self, scene) -> str:
        title = getattr(scene, "title", f"Scene {getattr(scene, 'id', '?')}")
        desc = getattr(scene, "description", "")
        narration = getattr(scene, "narration", "")
        base = f"{title}. {desc} {narration}".strip()

        style = (
            " high quality medical 3D render, surgical training, clear anatomy, "
            "soft studio lighting, professional, clean background, educational style, 4k, ultra detailed"
        )
        return base + style

    def _generate_image_via_diffusion(self, scene) -> str:
        """
        Hugging Face Inference API call for SDXL (router).

        Expected:
          DIFFUSION_API_URL = https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0
          DIFFUSION_API_KEY = hf_...  (or HF_TOKEN)
        """

        if not self.diffusion_url or not self.diffusion_key:
            raise RuntimeError("Diffusion config missing (url or key).")

        prompt = self._build_prompt(scene)

        payload = {
            "inputs": prompt,
            "options": {"wait_for_model": True},
        }

        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.diffusion_key}"

        # Retry strategy for transient failures (429/503/timeouts)
        # attempts: immediate, +2s, +5s
        last_err = None
        for attempt, sleep_s in enumerate([0, 2, 5], start=1):
            if sleep_s:
                time.sleep(sleep_s)

            try:
                resp = requests.post(
                    self.diffusion_url,
                    json=payload,
                    headers=headers,
                    timeout=300,
                )

                ctype = resp.headers.get("content-type")
                print("[HF STATUS]", resp.status_code, ctype)

                # Handle transient statuses
                if resp.status_code in (429, 503):
                    try:
                        print("[HF BODY]", resp.text[:500])
                    except Exception:
                        pass
                    last_err = RuntimeError(f"HF transient status {resp.status_code}")
                    continue

                # Any other non-OK => raise
                if not resp.ok:
                    try:
                        print("[HF BODY]", resp.text[:500])
                    except Exception:
                        pass
                    resp.raise_for_status()

                image_bytes = resp.content

                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError("Failed to decode image from diffusion API response")

                filename = f"scene_{getattr(scene, 'id', 'x')}_{uuid.uuid4().hex}.png"
                img_path = os.path.join(self.image_dir, filename)
                cv2.imwrite(img_path, img)

                print(f"[VanClient] SDXL image saved: {img_path}")
                return img_path

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
                print(f"[HF TIMEOUT] attempt {attempt}: {e}")
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break

        raise RuntimeError(f"Diffusion failed after retries: {last_err}")

    # ------------------------------------------------------------------
    # Image -> animated video with OpenCV
    # ------------------------------------------------------------------
    def _image_to_video(self, img_path: str, scene) -> str:
        img = cv2.imread(img_path)
        if img is None:
            return self._generate_text_demo_clip(scene)

        h, w, _ = img.shape
        scale = max(self.width / w, self.height / h) * 1.1
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        max_dx = max(0, new_w - self.width)
        max_dy = max(0, new_h - self.height)

        total_frames = self.fps * self.duration
        filename = f"scene_{getattr(scene, 'id', 'x')}_{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(self.video_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))

        if not writer.isOpened():
            return self._generate_text_demo_clip(scene)

        title_text = getattr(scene, "title", f"Scene {getattr(scene, 'id', '?')}")
        desc_text = getattr(scene, "description", "")[:120]

        for i in range(total_frames):
            t = i / max(total_frames - 1, 1)
            dx = int(max_dx * t * 0.7)
            dy = int(max_dy * t * 0.3)
            crop = resized[dy:dy + self.height, dx:dx + self.width].copy()

            overlay = crop.copy()
            bar_height = 90
            cv2.rectangle(overlay, (0, self.height - bar_height), (self.width, self.height), (0, 0, 0), -1)
            alpha = 0.55
            crop = cv2.addWeighted(overlay, alpha, crop, 1 - alpha, 0)

            cv2.putText(
                crop, title_text, (30, self.height - bar_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

            if desc_text:
                cv2.putText(
                    crop, desc_text, (30, self.height - bar_height + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA
                )

            progress = int(self.width * t)
            cv2.rectangle(crop, (0, self.height - 5), (progress, self.height), (0, 255, 255), -1)

            writer.write(crop)

        writer.release()
        return filename

    # ------------------------------------------------------------------
    # Pure OpenCV text demo (no diffusion)
    # ------------------------------------------------------------------
    def _generate_text_demo_clip(self, scene) -> str:
        total_frames = self.fps * self.duration
        filename = f"scene_{getattr(scene, 'id', 'x')}_{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(self.video_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))

        if not writer.isOpened():
            with open(filepath, "wb") as f:
                f.write(b"")
            return filename

        bg_color = (30, 30, 60)  # BGR
        accent_color = (120, 180, 255)
        text_color = (255, 255, 255)

        title_text = getattr(scene, "title", f"Scene {getattr(scene, 'id', '?')}")
        desc_text = getattr(scene, "description", "")[:140]

        for i in range(total_frames):
            t = i / max(total_frames - 1, 1)

            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = bg_color

            cv2.circle(frame, (self.width // 2, self.height // 2), int(self.width * 0.8), (40, 40, 90), -1)

            pad_x, pad_y = 60, 80
            cv2.rectangle(frame, (pad_x, pad_y), (self.width - pad_x, self.height - pad_y), (50, 50, 110), -1)
            cv2.rectangle(frame, (pad_x, pad_y), (self.width - pad_x, self.height - pad_y), accent_color, 2)

            cv2.putText(
                frame, title_text, (pad_x + 30, pad_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA
            )

            underline_len = int((self.width - pad_x * 2 - 60) * t)
            cv2.line(frame, (pad_x + 30, pad_y + 80), (pad_x + 30 + underline_len, pad_y + 80), accent_color, 3)

            if desc_text:
                max_chars = 50
                line1 = desc_text[:max_chars]
                line2 = desc_text[max_chars:max_chars * 2]

                cv2.putText(frame, line1, (pad_x + 30, pad_y + 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
                if line2:
                    cv2.putText(frame, line2, (pad_x + 30, pad_y + 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            progress = int((self.width - pad_x * 2) * t)
            cv2.rectangle(frame, (pad_x, self.height - pad_y + 10), (pad_x + progress, self.height - pad_y + 20),
                          accent_color, -1)

            writer.write(frame)

        writer.release()
        return filename
