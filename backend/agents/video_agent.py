from typing import List
from ..models.schemas import StoryboardItem, VideoClip
from ..services.van_client import VanClient

class VideoAgent:
    """
    Generates video clips for each scene using the VanClient.
    If a real video API is configured, actual MP4 files are written into
    backend/generated_videos and exposed via Flask with /videos/<filename> URLs.
    Otherwise, empty placeholder files are created so the pipeline is still testable.
    """
    def __init__(self):
        self.client = VanClient()

    def generate_videos(self, scenes: List[StoryboardItem]) -> List[VideoClip]:
        clips: List[VideoClip] = []
        for scene in scenes:
            filename = self.client.generate_clip(scene)
            video_url = f"/videos/{filename}"
            clips.append(VideoClip(
                id=scene.id,
                scene_title=scene.title,
                video_url=video_url
            ))
        return clips
