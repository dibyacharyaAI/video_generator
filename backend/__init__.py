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
        # Diffusion config (env first)
        # -----------------------------
        self.diffusion_url = os.getenv("DIFFUSION_API_URL")
        # local / codespaces me DIFFUSION_API_KEY ya HF_TOKEN se lo
        self.diffusion_key = os.getenv("DIFFUSION_API_KEY") or os.getenv("HF_TOKEN")
        self.diffusion_mode = os.getenv("DIFFUSION_API_MODE", "bytes").lower()  # "bytes" or "url"
        self.negative_prompt = os.getenv("DIFFUSION_NEGATIVE_PROMPT", "")

        # Agar env se nahi mila, to Streamlit Cloud secrets se try karo
        if (not self.diffusion_url or not self.diffusion_key) and st is not None:
            try:
                if not self.diffusion_url and "DIFFUSION_API_URL" in st.secrets:
                    self.diffusion_url = st.secrets["DIFFUSION_API_URL"]
                if not self.diffusion_key and "DIFFUSION_API_KEY" in st.secrets:
                    self.diffusion_key = st.secrets["DIFFUSION_API_KEY"]
            except Exception as e:
                print(f"[VanClient] Could not read Streamlit secrets: {e}")

        print(
            "[VanClient] init:",
            "diffusion_url =", self.diffusion_url,
            "| has_key =", bool(self.diffusion_key),
        )

        # Video settings
        self.width = int(os.getenv("VIDEO_WIDTH", "960"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "540"))
        self.fps = int(os.getenv("VIDEO_FPS", "24"))
        self.duration = int(os.getenv("VIDEO_CLIP_SECONDS", "6"))
