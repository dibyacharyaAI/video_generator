import json
import os
from typing import List

class RAGService:
    """Placeholder for retrieval augmented generation service."""
    def __init__(self):
        # Load some static medical facts from JSON
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'medical_facts.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            self.medical_data = json.load(f)

    def get_medical_context(self, prompt: str) -> List[str]:
        # Very naive retrieval: return all entries containing keywords from prompt
        keywords = [word.lower() for word in prompt.split() if len(word) > 3]
        results = []
        for entry in self.medical_data:
            if any(keyword in entry['topic'].lower() for keyword in keywords):
                results.append(entry['fact'])
        # If none matched, just return a generic explanation
        if not results and self.medical_data:
            results.append(self.medical_data[0]['fact'])
        return results
