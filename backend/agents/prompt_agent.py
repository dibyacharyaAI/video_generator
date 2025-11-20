
class PromptRefinementAgent:
    """Refines the raw user prompt by adding context and structure"""
    def refine(self, prompt: str) -> str:
        refined = prompt.strip()
        # Append step-by-step tutorial mention if missing
        if "tutorial" not in refined.lower():
            refined += " (step-by-step tutorial)"
        # Prepend surgeon perspective if not present
        if "surgeon" not in refined.lower():
            refined = "Surgeon perspective: " + refined
        return refined
