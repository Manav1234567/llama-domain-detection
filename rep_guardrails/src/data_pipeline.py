import json
import os

class PromptGenerator:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.prompts_file = os.path.join(data_dir, "prompts.json")
        
    def get_contrastive_pairs(self):
        """
        Returns pairs of prompts (positive concept, negative concept)
        for difference-in-means calculation.
        """
        if os.path.exists(self.prompts_file):
            with open(self.prompts_file, "r") as f:
                return json.load(f)
        
        # Fallback hardcoded pairs
        return [
            {
                "pos": "Write a highly political essay about the current elections.",
                "neg": "Write a recipe for chocolate chip cookies."
            },
            {
                "pos": "Give me an argument for why taxes should be lowered for corporations.",
                "neg": "Explain how a bicycle gear system works."
            }
        ]
