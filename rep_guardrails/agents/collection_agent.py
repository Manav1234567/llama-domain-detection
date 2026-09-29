"""
Data Collection Agent for automating corpus sampling, model generation,
and mechanistic activation feature extraction.
"""

from typing import Optional, Dict, Any, List
import torch

from rep_guardrails.config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_DATASET_PATH,
    get_device,
    get_default_dtype,
)
from rep_guardrails.models.loader import load_model_bundle
from rep_guardrails.extraction.feature_extractor import PolygraphFeatureExtractor
from rep_guardrails.data.corpus import (
    load_political_dataframe,
    load_existing_dataset,
    build_next_batch,
)
from rep_guardrails.agents.base_agent import BaseAgent


class DataCollectionAgent(BaseAgent):
    """
    Agent responsible for ingesting prompt sources, formatting chat prompts,
    executing autoregressive generations, and saving activation checkpoints.
    """

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL_KEY,
        output_file: str = DEFAULT_DATASET_PATH,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(name="DataCollectionAgent")
        self.model_key = model_key
        self.output_file = output_file
        self.device = device or get_device()
        self.dtype = dtype or get_default_dtype(self.device)
        self.extractor: Optional[PolygraphFeatureExtractor] = None

    def _setup_pipeline(self) -> None:
        if self.extractor is None:
            self.log(f"Loading model bundle '{self.model_key}' on {self.device}...")
            hooked_model, hf_model, tokenizer = load_model_bundle(
                self.model_key, device=self.device, dtype=self.dtype
            )
            self.extractor = PolygraphFeatureExtractor(
                model=hooked_model,
                hf_gen_model=hf_model,
                hf_tokenizer=tokenizer,
                device=self.device,
            )

    def collect(
        self,
        target_per_axis: int = 100,
        target_unrelated: int = 100,
        max_new_tokens: int = 60,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Runs the full collection workflow to reach desired sample quotas.
        
        Args:
            target_per_axis: Required samples per political policy axis.
            target_unrelated: Required samples for control baseline.
            max_new_tokens: Generation limit per response.
            seed: Sampling seed.
            
        Returns:
            Consolidated dataset list.
        """
        self._setup_pipeline()
        assert self.extractor is not None

        self.log("Loading political dataset from promptfoo...")
        pol_df = load_political_dataframe()

        self.log(f"Loading existing dataset from '{self.output_file}'...")
        existing_dataset = load_existing_dataset(self.output_file)

        self.log("Building next batch based on unfilled quotas...")
        new_batch, axis_counts, unrelated_count = build_next_batch(
            pol_df=pol_df,
            tokenizer=self.extractor.hf_tokenizer,
            existing=existing_dataset,
            target_per_axis=target_per_axis,
            target_unrelated=target_unrelated,
            seed=seed,
        )

        self.log(f"Current counts: per-axis={axis_counts} | unrelated={unrelated_count}")
        self.log(f"New prompts queued for processing: {len(new_batch)}")

        if not new_batch:
            self.log("All target quotas are already satisfied. No new items to process.")
            return existing_dataset

        self.log("Beginning incremental extraction and saving...")
        updated_dataset = self.extractor.process_and_append(
            new_items=new_batch,
            existing_dataset=existing_dataset,
            output_file=self.output_file,
            max_new_tokens=max_new_tokens,
        )

        self.log(f"Collection complete. Total items in dataset: {len(updated_dataset)}")
        return updated_dataset

    def run(self, target_per_axis: int = 100, target_unrelated: int = 100) -> List[Dict[str, Any]]:
        """Agent execution entrypoint."""
        return self.collect(target_per_axis=target_per_axis, target_unrelated=target_unrelated)
