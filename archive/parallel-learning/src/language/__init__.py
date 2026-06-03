"""
语言领域：符号接地 + 语言涌现 + 语法 + 通信协议 + 语用学 + 记忆支架 + 连续概念
         + 内部言语 + 叙事 + 跨模态接地 + 语言引导注意力
"""

from src.language.emergence import EmergingLanguage
from src.language.grounding import GroundingModule
from src.language.grammar import GrammarSystem, GrammarRule
from src.language.communication import CommunicationProtocol
from src.language.pragmatics import PragmaticsModule, SocialContext, CONTEXT_MARKERS
from src.language.memory_scaffold import DualCodingMemory
from src.language.continuous_concepts import ContinuousConceptSpace
from src.language.inner_speech import InnerSpeechModule
from src.language.narrative import (
    Event, Narrative, NarrativeModule,
    NARRATIVE_MARKERS, TEMPORAL_MARKERS, CAUSAL_MARKERS,
)
from src.language.crossmodal import (
    CrossModalObject, CrossModalGrounding, MODALITY_TYPES,
)
from src.language.attention import AttentionModulator
