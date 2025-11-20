from typing import List
from ..models.schemas import StoryboardItem

class ScriptAgent:
    """Generates narration text for each storyboard scene."""
    def generate_scripts(self, scenes: List[StoryboardItem]) -> List[StoryboardItem]:
        for scene in scenes:
            # A simple narration generator; in real system call to LLM like Claude
            scene.narration = f"This scene explains: {scene.description}"
        return scenes
