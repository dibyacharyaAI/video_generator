from typing import List, Dict
from ..models.schemas import StoryboardItem

class DirectorAgent:
    """Breaks a prompt into scenes using retrieved context."""
    def __init__(self, rag_service):
        self.rag_service = rag_service

    def plan(self, prompt: str, context: List[str]) -> List[StoryboardItem]:
        # For demonstration, create simple scenes based on prompt and context
        # In reality, this would involve complex planning using LLM
        scenes = []
        # Example: split context lines into scenes
        if not context:
            # fallback: create one scene with prompt summary
            scenes.append(StoryboardItem(id=1, title="Overview", description=prompt, narration=""))
            return scenes

        for idx, info in enumerate(context):
            scenes.append(StoryboardItem(
                id=idx+1,
                title=f"Scene {idx+1}",
                description=info,
                narration=""
            ))
        return scenes
