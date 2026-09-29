"""
Ground Truth Judge Agent for executing automated Zero-Shot NLI response labeling.
"""

from typing import Optional, List, Dict, Any
import os
import joblib
import torch

from rep_guardrails.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_JUDGE_MODEL,
    get_device,
)
from rep_guardrails.data.judge import PoliticalJudge
from rep_guardrails.data.corpus import load_existing_dataset
from rep_guardrails.agents.base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """
    Agent automating the 'Response-Labeling Paradigm' via DeBERTa zero-shot NLI.
    Ensures linear probes are trained on verified cognitive outputs rather than assumed intents.
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        model_name: str = DEFAULT_JUDGE_MODEL,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(name="JudgeAgent")
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.device = device or get_device()
        self.judge: Optional[PoliticalJudge] = None

    def _setup_judge(self) -> None:
        if self.judge is None:
            self.log(f"Initializing PoliticalJudge with model '{self.model_name}' on {self.device}...")
            self.judge = PoliticalJudge(model_name=self.model_name, device=self.device)

    def score(
        self,
        dataset_path: Optional[str] = None,
        save_every: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Loads dataset, scores all responses for political relevance, and saves to disk.
        
        Args:
            dataset_path: Path to dataset file (defaults to configured path).
            save_every: Checkpoint frequency.
            
        Returns:
            Enriched dataset list with 'response_political_relevance' scores.
        """
        target_path = dataset_path or self.dataset_path
        self.log(f"Loading dataset from '{target_path}'...")
        dataset = load_existing_dataset(target_path)
        if not dataset:
            raise FileNotFoundError(f"No valid dataset entries found at '{target_path}'.")

        self._setup_judge()
        assert self.judge is not None

        self.log(f"Scoring {len(dataset)} responses with zero-shot NLI...")
        enriched = self.judge.score_dataset(
            dataset=dataset,
            output_file=target_path,
            save_every=save_every,
        )

        self.log(f"Annotation complete. Saved enriched dataset to '{target_path}'.")
        return enriched

    def run(self, dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Agent entrypoint matching BaseAgent interface."""
        return self.score(dataset_path=dataset_path)
