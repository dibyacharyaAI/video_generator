from typing import List, Optional
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

class StoryboardItem(BaseModel):
    id: int
    title: str
    description: str
    narration: Optional[str] = None

class StoryboardResponse(BaseModel):
    scenes: List[StoryboardItem]

class VideoClip(BaseModel):
    id: int
    scene_title: str
    video_url: str

class VideoResponse(BaseModel):
    clips: List[VideoClip]

class ReviewResponse(BaseModel):
    score: float
    comments: str
