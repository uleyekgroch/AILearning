"""假设-验证-修正引擎 — 科学方法式的主动学习"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.entity import Entity
from src.knowledge.relation import Relation, REL_IS_A, REL_HAS_PROPERTY, REL_CAUSES


class HypothesisStatus(Enum):
    PROPOSED = 'proposed'
    TESTING = 'testing'
    CONFIRMED = 'confirmed'
    REFUTED = 'refuted'
    REVISED = 'revised'


@dataclass
class Hypothesis:
    id: str
    claim: str
    scope: Dict[str, Any] = field(default_factory=dict)
    evidence_for: List[Dict[str, Any]] = field(default_factory=list)
    evidence_against: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    parent_id: Optional[str] = None
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    tested_count: int = 0

    @property
    def evidence_ratio(self) -> float:
        total = len(self.evidence_for) + len(self.evidence_against)
        if total == 0:
            return 0.5
        return len(self.evidence_for) / total

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'claim': self.claim,
            'scope': self.scope,
            'evidence_for': self.evidence_for,
            'evidence_against': self.evidence_against,
            'confidence': self.confidence,
            'status': self.status.value,
            'parent_id': self.parent_id,
            'predictions': self.predictions,
            'created_at': self.created_at,
            'tested_count': self.tested_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Hypothesis':
        return cls(
            id=d['id'],
            claim=d['claim'],
            scope=d.get('scope', {}),
            evidence_for=d.get('evidence_for', []),
            evidence_against=d.get('evidence_against', []),
            confidence=d.get('confidence', 0.5),
            status=HypothesisStatus(d.get('status', 'proposed')),
            parent_id=d.get('parent_id'),
            predictions=d.get('predictions', []),
            created_at=d.get('created_at', 0),
            tested_count=d.get('tested_count', 0),
        )


class HypothesisEngine:
    """假设引擎：提出假设、设计验证、确认/反驳/修正

    核心循环：观察 → 假设 → 预测 → 验证 → 修正
    确认的假设写入 KnowledgeGraph。
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.hypotheses: Dict[str, Hypothesis] = {}
        self._next_id = 0

    def _make_id(self) -> str:
        self._next_id += 1
        return f'hyp:{self._next_id}'

    def propose_hypothesis(self, observations: List[Dict],
                           claim: Optional[str] = None) -> Hypothesis:
        """从观察中提出假设（奥卡姆剃刀：选最简解释）"""
        if not observations:
            raise ValueError("observations 不能为空")

        if claim is None:
            claim = self._generate_claim(observations)

        hyp = Hypothesis(
            id=self._make_id(),
            claim=claim,
            scope=self._extract_scope(observations),
            confidence=0.5,
            status=HypothesisStatus.PROPOSED,
            predictions=self._generate_predictions(claim, observations),
        )
        self.hypotheses[hyp.id] = hyp
        return hyp

    def design_test(self, hypothesis_id: str) -> Dict[str, Any]:
        """为假设设计验证方案"""
        hyp = self.hypotheses.get(hypothesis_id)
        if hyp is None:
            return {'error': f'假设 {hypothesis_id} 不存在'}

        hyp.status = HypothesisStatus.TESTING

        test = {
            'hypothesis_id': hypothesis_id,
            'claim': hyp.claim,
            'predictions': hyp.predictions,
            'test_type': self._choose_test_type(hyp),
            'required_observations': [],
        }

        for pred in hyp.predictions:
            test['required_observations'].append({
                'prediction': pred,
                'expected': pred.get('expected'),
                'how_to_observe': pred.get('observation_hint', '直接观察'),
            })

        return test

    def test_hypothesis(self, hypothesis_id: str,
                        test_case: Dict[str, Any]) -> HypothesisStatus:
        """用测试用例验证假设，返回新状态"""
        hyp = self.hypotheses.get(hypothesis_id)
        if hyp is None:
            raise ValueError(f'假设 {hypothesis_id} 不存在')

        hyp.tested_count += 1
        prediction = test_case.get('prediction', {})
        actual = test_case.get('actual')
        expected = prediction.get('expected') if prediction else None

        evidence = {
            'prediction': prediction,
            'actual': actual,
            'expected': expected,
            'match': actual == expected if expected is not None else None,
            'test_number': hyp.tested_count,
        }

        if expected is not None and actual == expected:
            hyp.evidence_for.append(evidence)
        elif expected is not None:
            hyp.evidence_against.append(evidence)

        hyp.confidence = self._update_confidence(hyp)

        if hyp.confidence >= 0.8 and len(hyp.evidence_for) >= 2:
            hyp.status = HypothesisStatus.CONFIRMED
            self._write_to_graph(hyp)
        elif hyp.confidence <= 0.2 and len(hyp.evidence_against) >= 2:
            hyp.status = HypothesisStatus.REFUTED

        return hyp.status

    def revise_hypothesis(self, hypothesis_id: str,
                          new_claim: str) -> Hypothesis:
        """基于反驳结果修正假设，创建新版本"""
        old = self.hypotheses.get(hypothesis_id)
        if old is None:
            raise ValueError(f'假设 {hypothesis_id} 不存在')

        revised = Hypothesis(
            id=self._make_id(),
            claim=new_claim,
            scope=dict(old.scope),
            evidence_for=list(old.evidence_for),
            confidence=0.4,
            status=HypothesisStatus.PROPOSED,
            parent_id=old.id,
        )
        self.hypotheses[revised.id] = revised
        return revised

    def get_hypothesis_history(self, hypothesis_id: str) -> List[Hypothesis]:
        """获取假设的修正链（从最初到当前）"""
        chain = []
        current = self.hypotheses.get(hypothesis_id)
        if current is None:
            return chain
        chain.append(current)
        while current.parent_id:
            current = self.hypotheses.get(current.parent_id)
            if current is None:
                break
            chain.append(current)
        chain.reverse()
        return chain

    def get_active_hypotheses(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses.values()
                if h.status in (HypothesisStatus.PROPOSED,
                                HypothesisStatus.TESTING)]

    def get_confirmed_hypotheses(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses.values()
                if h.status == HypothesisStatus.CONFIRMED]

    # ── 持久化 ────────────────────────────────────────────────

    def save_state(self) -> dict:
        return {
            'hypotheses': {hid: h.to_dict() for hid, h in self.hypotheses.items()},
            'next_id': self._next_id,
        }

    def load_state(self, state: dict) -> None:
        self.hypotheses.clear()
        for hid, hd in state.get('hypotheses', {}).items():
            self.hypotheses[hid] = Hypothesis.from_dict(hd)
        self._next_id = state.get('next_id', 0)

    # ── 内部方法 ──────────────────────────────────────────────

    def _generate_claim(self, observations: List[Dict]) -> str:
        """从观察中生成最简假设"""
        if len(observations) == 1:
            obs = observations[0]
            if 'property' in obs and 'value' in obs:
                return f"具有 {obs['property']}={obs['value']} 的事物具有该属性"
            if 'action' in obs:
                return f"在 {obs.get('context', '某种')} 条件下会发生 {obs['action']}"
            return f"观察到 {obs}"

        shared = self._find_commonalities(observations)
        if shared:
            parts = [f"{k}={v}" for k, v in shared.items()]
            return f"共同特征: {', '.join(parts)}"

        return f"基于 {len(observations)} 次观察的模式假设"

    def _extract_scope(self, observations: List[Dict]) -> Dict:
        scope = {}
        for obs in observations:
            for key in ('domain', 'context', 'entity_type'):
                if key in obs:
                    scope[key] = obs[key]
        return scope

    def _generate_predictions(self, claim: str,
                              observations: List[Dict]) -> List[Dict]:
        """基于假设生成可验证的预测"""
        preds = []
        for obs in observations:
            pred = {
                'claim': claim,
                'expected': obs,
                'observation_hint': '重复观察以验证一致性',
            }
            preds.append(pred)
        return preds

    def _choose_test_type(self, hyp: Hypothesis) -> str:
        if '因果关系' in hyp.claim or '导致' in hyp.claim:
            return 'causal_test'
        if '共同特征' in hyp.claim or '属性' in hyp.claim:
            return 'property_check'
        return 'observation_repeat'

    def _update_confidence(self, hyp: Hypothesis) -> float:
        total = len(hyp.evidence_for) + len(hyp.evidence_against)
        if total == 0:
            return hyp.confidence
        ratio = hyp.evidence_ratio
        weight = min(1.0, total / 5.0)
        prior = 0.5
        return prior * (1 - weight) + ratio * weight

    def _write_to_graph(self, hyp: Hypothesis) -> None:
        """将已确认的假设写入知识图谱"""
        entity = Entity(
            id=f'hyp_confirmed:{hyp.id}',
            type='confirmed_hypothesis',
            properties={
                'claim': hyp.claim,
                'evidence_count': len(hyp.evidence_for),
                'tested_count': hyp.tested_count,
            },
            confidence=hyp.confidence,
            source='hypothesis',
            tags=['confirmed'],
        )
        self.graph.add_entity(entity)

        for scope_key, scope_val in hyp.scope.items():
            scope_entity = self.graph.get_or_create(
                f'concept:{scope_val}', 'concept',
                properties={'name': scope_val},
            )
            self.graph.add_relation(Relation(
                source_id=entity.id,
                target_id=scope_entity.id,
                type=REL_HAS_PROPERTY,
                confidence=hyp.confidence,
                evidence_count=len(hyp.evidence_for),
            ))

    def _find_commonalities(self, observations: List[Dict]) -> Dict:
        if not observations:
            return {}
        common = dict(observations[0])
        for obs in observations[1:]:
            keys_to_remove = [k for k in common if common.get(k) != obs.get(k)]
            for k in keys_to_remove:
                del common[k]
        return common
