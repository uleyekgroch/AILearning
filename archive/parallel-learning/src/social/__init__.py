"""
社会领域：多 Agent 管理 + 交互协议 + 社会规范
"""

from src.social.adversarial import (
    TrustEvaluator, ReputationSystem, AdversarialModule, TRUST_MARKERS,
)
from src.social.planning import (
    CooperativePlanner, CooperativeTask, TaskStep,
    ROLE_MARKERS, SEQUENCE_MARKERS,
)
