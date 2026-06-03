"""Phase 71: 空间关系语言 — 介词涌现（左/右/上/下）

当场景中多个同色同形物体使纯属性描述失效时，
空间介词（left/right/above/below/near）从歧义消解压力中涌现。

4 个实验：
1. 介词涌现：追踪 spatial marker 随轮次增长
2. 空间 vs 属性：歧义场景中空间关系的优势
3. 参考框架：egocentric vs allocentric 收敛对比
4. 推理链："A left B left C" 传递性推断
"""
import json, random, numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from language_emergence import EmergingLanguage, _symbol_category, COLORS, SHAPES

SPATIAL_MARKERS = {'left', 'right', 'above', 'below', 'near'}
INV_REL = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above', 'near': 'near'}


# ── SpatialScene ──────────────────────────────────────────────

class SpatialScene:
    """带 (x,y) 坐标的 2D 网格场景"""

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size

    def generate(self, num_objects: int = 6, grid_size: int = 10,
                 force_ambiguity: bool = True) -> List[Dict]:
        colors, shapes = list(COLORS), list(SHAPES)
        positions, objects = set(), []
        if force_ambiguity:
            dc, ds = random.choice(colors), random.choice(shapes)
            for i in range(num_objects):
                c, s = (dc, ds) if i < 2 else (random.choice(colors), random.choice(shapes))
                x, y = self._pos(positions, grid_size)
                objects.append({'color': c, 'shape': s, 'x': x, 'y': y})
        else:
            for _ in range(num_objects):
                x, y = self._pos(positions, grid_size)
                objects.append({'color': random.choice(colors),
                                'shape': random.choice(shapes), 'x': x, 'y': y})
        return objects

    @staticmethod
    def _pos(used: set, gs: int) -> Tuple[int, int]:
        for _ in range(200):
            p = (random.randint(0, gs - 1), random.randint(0, gs - 1))
            if p not in used:
                used.add(p)
                return p
        return random.randint(0, gs - 1), random.randint(0, gs - 1)

    @staticmethod
    def get_spatial_relations(a: Dict, b: Dict, near_thr: float = 2.0) -> str:
        dx, dy = a['x'] - b['x'], a['y'] - b['y']
        if np.sqrt(dx**2 + dy**2) < near_thr:
            return 'near'
        if abs(dx) >= abs(dy):
            return 'left' if dx < 0 else 'right'
        return 'below' if dy < 0 else 'above'


# ── SpatialSpeaker ────────────────────────────────────────────

class SpatialSpeaker:
    """先 color+shape，冲突时附加 "A SPATIAL B" 消歧"""

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.spatial_markers: Dict[str, Dict] = {
            m: {'frequency': 0, 'successes': 0} for m in SPATIAL_MARKERS}
        self.spatial_used = 0

    def describe(self, target_idx: int, scene: List[Dict]) -> List[str]:
        target = scene[target_idx]
        utt = [target['color'], target['shape']]
        conflicts = [i for i, o in enumerate(scene)
                     if i != target_idx and o['color'] == target['color']
                     and o['shape'] == target['shape']]
        if conflicts:
            ref = min(conflicts, key=lambda i: abs(scene[i]['x'] - target['x'])
                      + abs(scene[i]['y'] - target['y']))
            rel = SpatialScene.get_spatial_relations(target, scene[ref])
            self.spatial_markers[rel]['frequency'] += 1
            self.spatial_used += 1
            utt += [rel, scene[ref]['color'], scene[ref]['shape']]
        return utt

    def record_success(self, utterance: List[str], success: bool):
        for sym in utterance:
            if sym in self.spatial_markers and success:
                self.spatial_markers[sym]['successes'] += 1


# ── SpatialListener ───────────────────────────────────────────

class SpatialListener:
    """解析 "color shape [SPATIAL ref_color ref_shape]" 话语"""

    def __init__(self, language: EmergingLanguage):
        self.language = language

    def interpret(self, utterance: List[str], scene: List[Dict]) -> Optional[int]:
        if len(utterance) < 2:
            return None
        tc, ts = utterance[0], utterance[1]
        cands = [i for i, o in enumerate(scene) if o['color'] == tc and o['shape'] == ts]
        if not cands:
            cands = [i for i, o in enumerate(scene) if o['color'] == tc]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # 空间消歧
        if len(utterance) >= 5 and utterance[2] in SPATIAL_MARKERS:
            marker, rc, rs = utterance[2], utterance[3], utterance[4]
            refs = [i for i, o in enumerate(scene) if o['color'] == rc and o['shape'] == rs]
            if refs:
                ref_obj = scene[refs[0]]
                matches = [c for c in cands
                           if SpatialScene.get_spatial_relations(scene[c], ref_obj) == marker]
                if matches:
                    return matches[0]
        return cands[0]


# ── Games ─────────────────────────────────────────────────────

class BaselineSpatialGame:
    """基线：只用 color+shape，多候选时随机选"""

    def __init__(self, language: EmergingLanguage):
        self.language = language
        self.games_played = self.successes = 0

    def play_round(self, scene: List[Dict], target_idx: int) -> bool:
        self.games_played += 1
        target = scene[target_idx]
        utt = [target['color'], target['shape']]
        cands = [i for i, o in enumerate(scene)
                 if o['color'] == target['color'] and o['shape'] == target['shape']]
        if not cands:
            cands = [i for i, o in enumerate(scene) if o['color'] == target['color']]
        chosen = random.choice(cands) if cands else None
        success = chosen == target_idx
        if success:
            self.successes += 1
        self.language.record_usage(utt, success)
        return success


class SpatialGame:
    """完整空间关系游戏，追踪介词涌现"""

    def __init__(self, language: EmergingLanguage, scene_gen: Optional[SpatialScene] = None):
        self.language = language
        self.speaker = SpatialSpeaker(language)
        self.listener = SpatialListener(language)
        self.scene_gen = scene_gen or SpatialScene()
        self.games_played = self.successes = 0
        self.ambiguous_games = self.spatial_successes = 0

    def play_round(self) -> Dict:
        scene = self.scene_gen.generate(6, 10, force_ambiguity=True)
        target_idx = random.randint(0, len(scene) - 1)
        utt = self.speaker.describe(target_idx, scene)
        chosen = self.listener.interpret(utt, scene)
        success = chosen == target_idx
        self.games_played += 1
        if success:
            self.successes += 1
        self.speaker.record_success(utt, success)
        self.language.record_usage(utt, success)
        target = scene[target_idx]
        amb = any(o['color'] == target['color'] and o['shape'] == target['shape']
                  for i, o in enumerate(scene) if i != target_idx)
        if amb:
            self.ambiguous_games += 1
            if success and len(utt) >= 5:
                self.spatial_successes += 1
        return {'success': success, 'utterance': utt, 'ambiguous': amb}

    def get_spatial_stats(self) -> Dict:
        markers = {}
        for m, s in self.speaker.spatial_markers.items():
            f = s['frequency']
            markers[m] = {'frequency': f, 'successes': s['successes'],
                          'success_rate': round(s['successes'] / f, 4) if f else 0.0}
        return {'markers': markers, 'total_spatial_used': self.speaker.spatial_used,
                'ambiguous_games': self.ambiguous_games, 'spatial_successes': self.spatial_successes}


# ── 参考框架辅助 ─────────────────────────────────────────────

def _egocentric_relation(obj: Dict, viewer: Tuple[int, int] = (5, 0)) -> str:
    dx, dy = obj['x'] - viewer[0], obj['y'] - viewer[1]
    d = np.sqrt(dx**2 + dy**2)
    if d < 1.5:
        return 'near'
    if abs(dx) >= abs(dy):
        return 'right' if dx > 0 else 'left'
    return 'above' if dy > 0 else 'below'


# ── 推理链 ────────────────────────────────────────────────────

class SpatialChain:
    """"A left B" + "B left C" => 传递性推断 A-C"""

    def __init__(self):
        self.links: List[Dict] = []

    def add_link(self, a: int, rel: str, b: int):
        self.links.append({'a': a, 'rel': rel, 'b': b})

    def infer(self, a: int, c: int) -> Optional[str]:
        for l1 in self.links:
            if l1['a'] != a:
                continue
            for l2 in self.links:
                if l2['a'] == l1['b'] and l2['b'] == c:
                    inf = self._transitive(l1['rel'], l2['rel'])
                    if inf:
                        return inf
        return None

    @staticmethod
    def _transitive(r1: str, r2: str) -> Optional[str]:
        if r1 == r2 and r1 in ('left', 'right', 'above', 'below'):
            return r1
        if r1 in ('left', 'right') and r2 in ('above', 'below'):
            return r1
        return None


# ══════════════════════════════════════════════════════════════
# 实验 1：介词涌现
# ══════════════════════════════════════════════════════════════

def experiment_1_preposition_emergence(num_rounds=300):
    """追踪空间介词涌现，每 50 轮打快照"""
    print("=" * 60)
    print("实验 1: 空间介词涌现")
    print("=" * 60)
    lang = EmergingLanguage()
    game = SpatialGame(lang)
    snapshots = []
    for r in range(num_rounds):
        game.play_round()
        if (r + 1) % 50 == 0:
            st = game.get_spatial_stats()
            sr = game.successes / max(1, game.games_played)
            amb_sr = game.spatial_successes / max(1, game.ambiguous_games)
            snapshots.append({'round': r + 1, 'success_rate': round(sr, 4),
                              'ambiguous_sr': round(amb_sr, 4),
                              'spatial_used': st['total_spatial_used'],
                              'markers': st['markers']})
            ms = ', '.join(f"{m}={d['frequency']}/{d['success_rate']:.2f}"
                          for m, d in st['markers'].items() if d['frequency'] > 0)
            print(f"  Round {r+1}: SR={sr:.3f}, 歧义SR={amb_sr:.3f}, "
                  f"空间={st['total_spatial_used']}, [{ms}]")
    st = game.get_spatial_stats()
    return {'final_sr': round(game.successes / max(1, game.games_played), 4),
            'final_ambiguous_sr': round(game.spatial_successes / max(1, game.ambiguous_games), 4),
            'total_spatial_used': st['total_spatial_used'],
            'markers': st['markers'], 'snapshots': snapshots}


# ══════════════════════════════════════════════════════════════
# 实验 2：空间 vs 纯属性
# ══════════════════════════════════════════════════════════════

def experiment_2_spatial_vs_attribute(num_rounds=200, num_runs=5):
    """对比空间游戏 vs 基线（纯属性），多次运行取平均"""
    print("=" * 60)
    print("实验 2: 空间关系 vs 纯属性匹配")
    print("=" * 60)
    sp_srs, bl_srs, sp_amb, bl_amb = [], [], [], []
    for run in range(num_runs):
        random.seed(42 + run); np.random.seed(42 + run)
        # 空间
        g_s = SpatialGame(EmergingLanguage())
        a_suc, a_tot = 0, 0
        for _ in range(num_rounds):
            res = g_s.play_round()
            if res['ambiguous']:
                a_tot += 1
                if res['success']:
                    a_suc += 1
        sp_srs.append(g_s.successes / max(1, g_s.games_played))
        sp_amb.append(a_suc / max(1, a_tot))
        # 基线
        random.seed(42 + run); np.random.seed(42 + run)
        g_b = BaselineSpatialGame(EmergingLanguage())
        sg = SpatialScene()
        ba_suc, ba_tot = 0, 0
        for _ in range(num_rounds):
            sc = sg.generate(6, 10, force_ambiguity=True)
            ti = random.randint(0, len(sc) - 1)
            t = sc[ti]
            hc = any(o['color'] == t['color'] and o['shape'] == t['shape']
                     for i, o in enumerate(sc) if i != ti)
            ok = g_b.play_round(sc, ti)
            if hc:
                ba_tot += 1
                if ok:
                    ba_suc += 1
        bl_srs.append(g_b.successes / max(1, g_b.games_played))
        bl_amb.append(ba_suc / max(1, ba_tot))
        print(f"  Run {run+1}: 空间={sp_srs[-1]:.3f}, 基线={bl_srs[-1]:.3f}, "
              f"歧义空间={sp_amb[-1]:.3f}, 歧义基线={bl_amb[-1]:.3f}")
    asr, bsr = float(np.mean(sp_srs)), float(np.mean(bl_srs))
    asa, bsa = float(np.mean(sp_amb)), float(np.mean(bl_amb))
    imp = (asr - bsr) / max(.01, bsr) * 100
    aimp = (asa - bsa) / max(.01, bsa) * 100
    print(f"\n  总体: 空间={asr:.3f}, 基线={bsr:.3f}, 提升={imp:.1f}%")
    print(f"  歧义: 空间={asa:.3f}, 基线={bsa:.3f}, 提升={aimp:.1f}%")
    return {'spatial_sr': round(asr, 4), 'baseline_sr': round(bsr, 4),
            'improvement_pct': round(imp, 2),
            'spatial_ambiguous_sr': round(asa, 4),
            'baseline_ambiguous_sr': round(bsa, 4),
            'ambiguous_improvement_pct': round(aimp, 2)}


# ══════════════════════════════════════════════════════════════
# 实验 3：参考框架
# ══════════════════════════════════════════════════════════════

def experiment_3_reference_frame(num_rounds=200):
    """egocentric vs allocentric 参考框架收敛对比"""
    print("=" * 60)
    print("实验 3: 参考框架对比（自我中心 vs 他物中心）")
    print("=" * 60)
    # 他物中心 — 直接用 SpatialGame
    g_allo = SpatialGame(EmergingLanguage())
    for _ in range(num_rounds):
        g_allo.play_round()
    allo_sr = g_allo.successes / max(1, g_allo.games_played)
    # 自我中心
    lang_ego = EmergingLanguage()
    sg = SpatialScene()
    ego_succ, ego_tot, ego_used = 0, 0, 0
    ego_markers = defaultdict(lambda: [0, 0])  # [freq, succ]
    for _ in range(num_rounds):
        scene = sg.generate(6, 10, force_ambiguity=True)
        ti = random.randint(0, len(scene) - 1)
        target = scene[ti]
        utt = [target['color'], target['shape']]
        conflicts = [i for i, o in enumerate(scene) if i != ti
                     and o['color'] == target['color'] and o['shape'] == target['shape']]
        if conflicts:
            rel = _egocentric_relation(target)
            utt += [rel, 'viewer']
            ego_used += 1
            ego_markers[rel][0] += 1
        # 匹配
        cands = [i for i, o in enumerate(scene) if o['color'] == target['color']
                 and o['shape'] == target['shape']]
        if not cands:
            cands = [i for i, o in enumerate(scene) if o['color'] == target['color']]
        if len(cands) == 1:
            chosen = cands[0]
        elif len(utt) >= 3 and utt[2] in SPATIAL_MARKERS:
            matches = [c for c in cands if _egocentric_relation(scene[c]) == utt[2]]
            chosen = matches[0] if matches else cands[0]
        else:
            chosen = random.choice(cands) if cands else None
        success = chosen == ti
        ego_tot += 1
        if success:
            ego_succ += 1
            if len(utt) >= 3 and utt[2] in ego_markers:
                ego_markers[utt[2]][1] += 1
        lang_ego.record_usage(utt, success)
    ego_sr = ego_succ / max(1, ego_tot)
    winner = 'allocentric' if allo_sr >= ego_sr else 'egocentric'
    print(f"  他物中心 SR={allo_sr:.3f}, 自我中心 SR={ego_sr:.3f}")
    print(f"  更优框架: {winner}")
    ego_ms = {m: {'freq': v[0], 'sr': round(v[1] / max(1, v[0]), 4)} for m, v in ego_markers.items()}
    return {'allocentric_sr': round(allo_sr, 4), 'egocentric_sr': round(ego_sr, 4),
            'allocentric_used': g_allo.speaker.spatial_used,
            'egocentric_used': ego_used, 'winner': winner, 'ego_markers': ego_ms}


# ══════════════════════════════════════════════════════════════
# 实验 4：空间推理链
# ══════════════════════════════════════════════════════════════

def experiment_4_spatial_reasoning_chain(num_rounds=200):
    """"A left B" + "B left C" 传递性推断测试"""
    print("=" * 60)
    print("实验 4: 空间推理链")
    print("=" * 60)
    sg = SpatialScene(grid_size=20)
    chain = SpatialChain()
    # 训练：学习两两关系
    learned = 0
    for _ in range(num_rounds):
        sc = sg.generate(3, 20, force_ambiguity=False)
        colors_ok = len({o['color'] for o in sc}) == 3
        shapes_ok = len({o['shape'] for o in sc}) == 3
        if not (colors_ok and shapes_ok):
            continue
        for ai, bi in [(0, 1), (1, 2), (0, 2)]:
            rel = SpatialScene.get_spatial_relations(sc[ai], sc[bi])
            chain.add_link(ai, rel, bi)
            chain.add_link(bi, INV_REL[rel], ai)
            learned += 1
    # 推理测试
    tests, correct, covered = 0, 0, 0
    for _ in range(num_rounds):
        sc = sg.generate(4, 20, force_ambiguity=False)
        idx = random.sample(range(len(sc)), 3)
        oa, ob, oc = sc[idx[0]], sc[idx[1]], sc[idx[2]]
        rab = SpatialScene.get_spatial_relations(oa, ob)
        rbc = SpatialScene.get_spatial_relations(ob, oc)
        chain.add_link(0, rab, 1)
        chain.add_link(1, rbc, 2)
        inferred = chain.infer(0, 2)
        actual = SpatialScene.get_spatial_relations(oa, oc)
        tests += 1
        if inferred is not None:
            covered += 1
            if inferred == actual:
                correct += 1
    inf_rate = correct / max(1, tests)
    cov_rate = covered / max(1, tests)
    when_rate = correct / max(1, covered)
    print(f"  学习对={learned}, 推理测试={tests}, 正确率={inf_rate:.3f}, "
          f"覆盖率={cov_rate:.3f}, 推理时正确={when_rate:.3f}")
    # 空间游戏在推理链场景中
    lang = EmergingLanguage()
    game = SpatialGame(lang)
    cs, ct = 0, 0
    for _ in range(num_rounds):
        sc = sg.generate(5, 15, force_ambiguity=True)
        ti = random.randint(0, len(sc) - 1)
        utt = game.speaker.describe(ti, sc)
        chosen = game.listener.interpret(utt, sc)
        ok = chosen == ti
        game.speaker.record_success(utt, ok)
        lang.record_usage(utt, ok)
        ct += 1
        if ok:
            cs += 1
    print(f"  空间游戏(链场景) SR={cs / max(1, ct):.3f}")
    # 传递模式分布
    pats = defaultdict(int)
    for lk in chain.links:
        pats[lk['rel']] += 1
    total_links = max(1, len(chain.links))
    dist = {r: round(c / total_links, 4) for r, c in pats.items()}
    return {'learned_pairs': learned, 'inference_tests': tests,
            'inference_accuracy': round(inf_rate, 4),
            'inference_coverage': round(cov_rate, 4),
            'when_inferred_accuracy': round(when_rate, 4),
            'chain_game_sr': round(cs / max(1, ct), 4),
            'relation_distribution': dist}


# ── 主入口 ────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    random.seed(42)
    np.random.seed(42)
    results = {}
    results['experiment_1'] = experiment_1_preposition_emergence()
    results['experiment_2'] = experiment_2_spatial_vs_attribute()
    results['experiment_3'] = experiment_3_reference_frame()
    results['experiment_4'] = experiment_4_spatial_reasoning_chain()
    with open('spatial_language_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 spatial_language_results.json")
