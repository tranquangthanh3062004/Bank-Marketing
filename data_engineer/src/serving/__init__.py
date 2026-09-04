"""
Serving & Deployment module.
Provides Batch Lead Scoring and FastAPI REST service.
"""

from .batch_scorer import BatchLeadScorer
from .app import app

__all__ = [
    "BatchLeadScorer",
    "app",
]
