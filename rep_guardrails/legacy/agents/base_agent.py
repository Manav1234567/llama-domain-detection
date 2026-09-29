"""
Abstract base agent class defining common interface, logging,
and lifecycle management for autonomous workflow agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all operational agents in the representation guardrail system.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"rep_guardrails.agents.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Logs a formatted message via the agent's logger."""
        self.logger.log(level, message)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Primary execution method to be implemented by concrete agent classes."""
        raise NotImplementedError("Subclasses must implement run()")

    def benchmark_execution(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Executes a callable and measures elapsed execution time in milliseconds."""
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result, elapsed_ms
