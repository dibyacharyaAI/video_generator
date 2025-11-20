from typing import List, Dict
from ..models.schemas import StoryboardItem

class ReviewAgent:
    """Performs a basic storyboard quality review."""
    def review(self, scenes: List[StoryboardItem]) -> Dict[str, str]:
        # Basic scoring: number of scenes and presence of narration
        if not scenes:
            return {'score': 0.0, 'comments': 'No scenes generated.'}
        missing_narration = [scene.id for scene in scenes if not scene.narration]
        if missing_narration:
            return {'score': 0.5, 'comments': f'Scenes missing narration: {missing_narration}'}
        return {'score': 1.0, 'comments': 'Looks good'}
