"""Instrument think() to trace which path actually wins and why answers are low quality.

Logs every decision point in think() for 3 questions after learning 12 sentences.
"""

import sys
import os
import io
import traceback

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.core.config import LearnerConfig
from src.core.learner import Learner

# Collect all trace output here
trace_lines = []

def log(msg):
    print(msg)
    trace_lines.append(msg)


def main():
    log("=" * 80)
    log("THINK() EXECUTION PATH TRACER")
    log("=" * 80)

    # 1. Create learner
    log("\n[INIT] Creating Learner with default config...")
    config = LearnerConfig()
    learner = Learner(config)
    log(f"[INIT] Device: {learner.device}")
    log(f"[INIT] Stage: {learner.stage}")

    # 2. Learn 12 sentences
    training_texts = [
        '光合作用是植物利用阳光将二氧化碳转化为葡萄糖的过程',
        '光合作用在叶绿体中进行',
        '叶绿体是植物细胞中含有叶绿素的细胞器',
        '植物通过光合作用制造有机物',
        '阳光是光合作用的能量来源',
        '二氧化碳是光合作用的原料',
        '葡萄糖是光合作用的产物',
        '水是光合作用的必要条件',
        '叶绿素能够吸收阳光中的能量',
        '植物需要水分和阳光才能进行光合作用',
        '光合作用释放氧气',
        '氧气是光合作用的副产物',
    ]

    log(f"\n[LEARN] Learning {len(training_texts)} sentences...")
    for i, text in enumerate(training_texts):
        result = learner.learn_from_text(text)
        entities = result.get('entities', [])
        triples = result.get('triples', [])
        log(f"  [{i+1:2d}] {text[:30]}...")
        log(f"       entities: {entities[:8]}")
        log(f"       triples:  {len(triples)}")

    # 3. Check what's in the knowledge graph and concept space
    log("\n" + "=" * 80)
    log("[STATE] Knowledge Graph inspection")
    kg = learner.knowledge
    log(f"  KG entities ({len(kg.entities)}):")
    for eid, entity in list(kg.entities.items())[:20]:
        emb_info = f"emb={'Y' if hasattr(entity, 'embedding') and entity.embedding is not None else 'N'}"
        log(f"    {eid} ({emb_info})")

    # KG relations
    log(f"  KG relations:")
    for eid in list(kg.entities.keys())[:10]:
        rels = kg.get_relations_of(eid)
        for rel in rels[:3]:
            log(f"    {eid} --[{rel.type}]--> {rel.target_id} (conf={rel.confidence:.2f})")

    cs = learner._safe_registry_get('concept_space')
    if cs:
        log(f"\n[STATE] Concept Space: {len(cs.concepts)} concepts")
        for cid, node in list(cs.concepts.items())[:20]:
            log(f"    {cid}: freq={node.frequency}, strength={node.strength:.3f}, source={node.source}")

    # 4. Check registry for key modules
    log("\n" + "=" * 80)
    log("[REGISTRY] Checking key modules...")

    # simulation_reasoning via registry
    sr_via_registry = learner._safe_registry_get("simulation_reasoning")
    log(f"  simulation_reasoning via registry: {sr_via_registry}")
    # simulation_reasoning via property
    sr_via_prop = learner.simulation_reasoning
    log(f"  simulation_reasoning via property: {type(sr_via_prop).__name__}")

    # language_acquisition via registry
    la_via_registry = learner._safe_registry_get("language_acquisition")
    log(f"  language_acquisition via registry: {la_via_registry}")
    # language_acquisition via property
    la_via_prop = learner.language_acquisition
    log(f"  language_acquisition via property: {type(la_via_prop).__name__}")

    # concept_space
    cs_via_registry = learner._safe_registry_get("concept_space")
    log(f"  concept_space via registry: {type(cs_via_registry).__name__ if cs_via_registry else None}")

    # 5. Now trace think() for each question
    questions = [
        '光合作用在哪里进行',
        '什么是光合作用',
        '呼吸作用在哪里进行',
    ]

    for q_idx, question in enumerate(questions):
        log("\n" + "=" * 80)
        log(f"[THINK] Question {q_idx+1}: {question}")
        log("=" * 80)

        trace_think_internal(learner, question)

    # 6. Write output file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'think_trace.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(trace_lines))
    print(f"\n[OUTPUT] Written to {output_path}")


def trace_think_internal(learner, question):
    """Manually trace each path of think() with detailed logging."""

    import re

    # Encode question
    log(f"  [ENCODE] Encoding question...")
    question_repr = learner._encode_text(question)
    log(f"  [ENCODE] Shape: {question_repr.shape}")

    # --- TTT (Test-Time Training) ---
    skip_ttt = getattr(learner, '_skip_ttt', False)
    log(f"  [TTT] skip_ttt={skip_ttt}, has_learnable_encoder={hasattr(learner, '_learnable_encoder')}")

    # ===== PATH 0: SimulationReasoning via registry =====
    log("\n  --- PATH 0: SimulationReasoning via registry ---")
    sr = learner._safe_registry_get("simulation_reasoning")
    log(f"  [P0] sr from registry = {sr}")

    if sr:
        cs = learner._safe_registry_get("concept_space")
        context_concepts = []
        if cs and hasattr(cs, "activate"):
            try:
                activated_tmp = cs.activate(question, top_k=5, spread_depth=2)
                if activated_tmp:
                    context_concepts = [a.concept_id for a in activated_tmp[:5]]
                    log(f"  [P0] Activated concepts: {context_concepts}")
                    log(f"  [P0] Activation strengths: {[f'{a.activation:.3f}' for a in activated_tmp[:5]]}")
            except Exception as e:
                log(f"  [P0] Activation error: {e}")

        if context_concepts:
            try:
                sr_result = sr.reason(question, context_concepts)
                log(f"  [P0] sr.reason() confidence: {sr_result.confidence:.3f}")
                log(f"  [P0] sr.reason() reasoning_type: {sr_result.reasoning_type}")
                if sr_result.scene:
                    log(f"  [P0] Scene concepts: {sr_result.scene.concepts}")
                    log(f"  [P0] Scene confidence: {sr_result.scene.confidence:.3f}")
                if sr_result.causal_chains:
                    for chain in sr_result.causal_chains:
                        log(f"  [P0] Causal chain: {chain.steps} (conf={chain.confidence:.3f})")

                if sr_result.confidence >= 0.4:
                    log(f"  [P0] Confidence >= 0.4, trying compose/express...")
                    # Try compose
                    la = learner._safe_registry_get("language_acquisition")
                    answer = None
                    if la and hasattr(la, "compose"):
                        answer = la.compose(context_concepts, goal=question)
                        log(f"  [P0] la.compose() = '{answer}'")
                    if not answer or len(answer) < 5:
                        answer = sr.express(sr_result, question)
                        log(f"  [P0] sr.express() = '{answer}'")
                    if answer and len(answer) > 5:
                        log(f"  [P0] >>> WINNER: Path 0 returns: '{answer}'")
                    else:
                        log(f"  [P0] Answer too short or empty, falling through")
                else:
                    log(f"  [P0] Confidence {sr_result.confidence:.3f} < 0.4, skipping")
            except Exception as e:
                log(f"  [P0] sr.reason() error: {e}")
                traceback.print_exc()
    else:
        log(f"  [P0] sr is None (simulation_reasoning not in registry!) -> Path 0 always fails")

    # ===== PATH 0.5: LanguageAcquisition compose =====
    log("\n  --- PATH 0.5: LanguageAcquisition compose via registry ---")
    la = learner._safe_registry_get("language_acquisition")
    log(f"  [P0.5] la from registry = {la}")

    if la and hasattr(la, "compose"):
        cs = learner._safe_registry_get("concept_space")
        if cs and hasattr(cs, "activate"):
            try:
                act = cs.activate(question, top_k=3, spread_depth=1)
                if act:
                    log(f"  [P0.5] Activated: {[(a.concept_id, f'{a.activation:.3f}') for a in act]}")
                    if act[0].activation > 0.3:
                        concept_ids = [a.concept_id for a in act[:5]]
                        composed = la.compose(concept_ids, goal=question)
                        log(f"  [P0.5] compose({concept_ids}) = '{composed}'")
                        if composed and len(composed) > 10:
                            log(f"  [P0.5] >>> WINNER: Path 0.5 returns: '{composed}'")
                        else:
                            log(f"  [P0.5] Composed too short or empty, falling through")
                    else:
                        log(f"  [P0.5] Top activation {act[0].activation:.3f} <= 0.3, skipping")
            except Exception as e:
                log(f"  [P0.5] Error: {e}")
    else:
        log(f"  [P0.5] la is None or no compose -> Path 0.5 always fails")

    # ===== PATH 1: Concept Space Activation =====
    log("\n  --- PATH 1: Concept Space Activation Diffusion ---")
    activated = None
    try:
        cs = learner._safe_registry_get('concept_space')
        if cs:
            log(f"  [P1] concept_space has {len(cs.concepts)} concepts")
            if len(cs.concepts) >= 3:
                activated = cs.activate(question, top_k=5, spread_depth=2)
                if activated:
                    log(f"  [P1] Activated (top 5):")
                    for ac in activated[:5]:
                        log(f"  [P1]   {ac.concept_id}: activation={ac.activation:.3f}")
                    if activated[0].activation > 0.1:
                        answer = learner._synthesize_from_activation(activated, question)
                        log(f"  [P1] _synthesize_from_activation() = '{answer}'")
                        if answer and len(answer) > 10:
                            has_weak = bool(re.search(r'存在关联|相互关联|有一定关系', answer))
                            log(f"  [P1] Has weak phrases: {has_weak}")
                            if not has_weak:
                                log(f"  [P1] >>> WINNER: Path 1 returns: '{answer}'")
                            else:
                                log(f"  [P1] Weak phrases detected, falling through")
                        else:
                            log(f"  [P1] Answer too short or empty, falling through")
                    else:
                        log(f"  [P1] Top activation {activated[0].activation:.3f} <= 0.1, skipping")
                else:
                    log(f"  [P1] No activation results")
            else:
                log(f"  [P1] Not enough concepts ({len(cs.concepts)} < 3)")
        else:
            log(f"  [P1] concept_space is None")
    except Exception as e:
        log(f"  [P1] Error: {e}")
        traceback.print_exc()

    # ===== PATH 1.3: Simulation Reasoning Direct =====
    log("\n  --- PATH 1.3: SimulationReasoning direct (via property) ---")
    try:
        if activated:
            activated_labels = [a.concept_id for a in activated[:5]]
            log(f"  [P1.3] activated_labels: {activated_labels}")
            sr_result = learner.simulation_reasoning.reason(question, activated_labels)
            log(f"  [P1.3] confidence: {sr_result.confidence:.3f}")
            if sr_result.confidence >= 0.3:
                sr_answer = learner.simulation_reasoning.express(sr_result, question)
                log(f"  [P1.3] express() = '{sr_answer}'")
                if sr_answer and len(sr_answer) > 5:
                    log(f"  [P1.3] >>> WINNER: Path 1.3 returns: '{sr_answer}'")
                else:
                    log(f"  [P1.3] Answer too short, falling through")
            else:
                log(f"  [P1.3] Confidence {sr_result.confidence:.3f} < 0.3, skipping")
        else:
            log(f"  [P1.3] No activated concepts, skipping")
    except Exception as e:
        log(f"  [P1.3] Error: {e}")
        traceback.print_exc()

    # ===== PATH 1.5: Multi-hop =====
    log("\n  --- PATH 1.5: Multi-hop Reasoning ---")
    try:
        cs = learner._safe_registry_get('concept_space')
        if cs and activated:
            multi_hop_answer = learner._synthesize_multihop(cs, activated, question)
            log(f"  [P1.5] _synthesize_multihop() = '{multi_hop_answer}'")
            if multi_hop_answer and len(multi_hop_answer) > 10:
                log(f"  [P1.5] >>> WINNER: Path 1.5 returns: '{multi_hop_answer}'")
            else:
                log(f"  [P1.5] Answer too short or empty, falling through")
        else:
            log(f"  [P1.5] No concept_space or no activated concepts, skipping")
    except Exception as e:
        log(f"  [P1.5] Error: {e}")
        traceback.print_exc()

    # ===== PATH 2: Unified Reasoning Engine =====
    log("\n  --- PATH 2: Unified Reasoning Engine ---")
    try:
        if not hasattr(learner, '_reasoning_engine'):
            from src.reasoning.unified_engine import UnifiedReasoningEngine
            learner._reasoning_engine = UnifiedReasoningEngine(learner)

        reasoning_results = learner._reasoning_engine.reason(question)
        log(f"  [P2] Got {len(reasoning_results)} reasoning results")
        for r in reasoning_results[:5]:
            log(f"  [P2]   method={r.method}, conf={r.confidence:.3f}, content='{r.content[:60]}'")

        if reasoning_results:
            synthesized = learner._synthesize_from_reasoning(reasoning_results, question)
            log(f"  [P2] _synthesize_from_reasoning() = '{synthesized}'")
            if synthesized and synthesized != "我没有足够的信息来回答这个问题。":
                log(f"  [P2] >>> WINNER: Path 2 returns: '{synthesized}'")
            else:
                log(f"  [P2] No useful answer from reasoning engine")
        else:
            log(f"  [P2] No reasoning results at all")
    except Exception as e:
        log(f"  [P2] Error: {e}")
        traceback.print_exc()

    # ===== PATH 3: Analogy =====
    log("\n  --- PATH 3: Analogy Generalization ---")
    answer = None
    try:
        cs = learner._safe_registry_get("concept_space")
        if cs and hasattr(cs, "analogy") and hasattr(cs, "activate"):
            q_activated = cs.activate(question, top_k=3, spread_depth=1)
            if q_activated and len(q_activated) >= 2:
                log(f"  [P3] Analogy concepts: {[(a.concept_id, f'{a.activation:.3f}') for a in q_activated[:3]]}")
                a, b = q_activated[0].concept_id, q_activated[1].concept_id
                for ac in q_activated[:2]:
                    c = ac.concept_id
                    analogies = cs.analogy(a, b, c, top_k=3)
                    if analogies:
                        log(f"  [P3] Analogies from {c}: {analogies[:3]}")
            else:
                log(f"  [P3] Not enough activated concepts for analogy")
        else:
            log(f"  [P3] No concept_space or no analogy method")
    except Exception as e:
        log(f"  [P3] Error: {e}")
        traceback.print_exc()

    # ===== ACTUAL think() call =====
    log("\n  --- ACTUAL think() result ---")
    actual_answer = learner.think(question)
    log(f"  [ACTUAL] Answer: '{actual_answer}'")

    # Determine which path won by comparing
    log(f"\n  [ANALYSIS] The actual think() returned the same as one of the paths above.")
    log(f"  [ANALYSIS] Check which path's output matches the ACTUAL answer.")


if __name__ == '__main__':
    main()
