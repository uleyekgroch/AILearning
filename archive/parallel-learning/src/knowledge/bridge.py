"""语言-图谱桥接 — 将 EmergingLanguage 的词汇数据同步到 KnowledgeGraph"""

from typing import TYPE_CHECKING

from src.knowledge.entity import Entity
from src.knowledge.relation import Relation, REL_COLLOCATES, REL_SIMILAR_TO

if TYPE_CHECKING:
    from src.language.emergence import EmergingLanguage
    from src.language.grounding import GroundingModule


class LanguageGraphBridge:
    """桥接 EmergingLanguage 词汇系统与 KnowledgeGraph

    将扁平的词汇统计转换为结构化的知识图谱。
    """

    def __init__(self, graph: 'KnowledgeGraph'):
        self.graph = graph
        self._synced_vocab: set = set()

    def sync_vocabulary(self, language: 'EmergingLanguage') -> int:
        """同步词汇表 → 知识图谱实体

        Returns: 新增实体数
        """
        new_count = 0
        for word, data in language.vocabulary.items():
            if word not in self._synced_vocab:
                entity = Entity(
                    id=f'sym:{word}',
                    type='symbol',
                    properties={
                        'text': word,
                        'frequency': data.get('frequency', 0),
                        'successes': data.get('successes', 0),
                        'success_rate': data.get('success_rate', 0.0),
                        'exposures': data.get('exposures', 0),
                    },
                    confidence=data.get('success_rate', 0.0),
                    source='vocabulary',
                )
                self.graph.add_entity(entity)
                self._synced_vocab.add(word)
                new_count += 1
            else:
                # 更新属性
                entity = self.graph.get_entity(f'sym:{word}')
                if entity:
                    entity.properties.update({
                        'frequency': data.get('frequency', 0),
                        'successes': data.get('successes', 0),
                        'success_rate': data.get('success_rate', 0.0),
                        'exposures': data.get('exposures', 0),
                    })
                    entity.confidence = data.get('success_rate', 0.0)
        return new_count

    def sync_collocations(self, language: 'EmergingLanguage') -> int:
        """同步搭配关系 → 知识图谱关系

        Returns: 新增关系数
        """
        new_count = 0
        for (word_a, word_b), data in language.collocations.items():
            src_id = f'sym:{word_a}'
            tgt_id = f'sym:{word_b}'
            # 确保实体存在
            self.graph.get_or_create(src_id, 'symbol')
            self.graph.get_or_create(tgt_id, 'symbol')

            rel = Relation(
                source_id=src_id,
                target_id=tgt_id,
                type=REL_COLLOCATES,
                confidence=data.get('success_rate', 0.5),
                evidence_count=data.get('count', 0),
                metadata={'data': data},
            )
            # 检查是否已存在
            existing = [r for r in self.graph.get_relations_of(src_id, 'out')
                        if r.target_id == tgt_id and r.type == REL_COLLOCATES]
            if not existing:
                self.graph.add_relation(rel)
                new_count += 1
        return new_count

    def sync_grounding(self, grounding: 'GroundingModule') -> int:
        """同步符号接地 → 知识图谱实体的 embedding

        Returns: 同步的实体数
        """
        count = 0
        for symbol, mapping in grounding.symbol_mappings.items():
            entity_id = f'sym:{symbol}'
            entity = self.graph.get_entity(entity_id)
            if entity is None:
                entity = Entity(
                    id=entity_id, type='symbol',
                    properties={'text': symbol},
                    source='grounding',
                )
                self.graph.add_entity(entity)

            # 关联感知聚类信息
            if isinstance(mapping, dict):
                entity.properties['grounded'] = True
                entity.properties['confidence'] = mapping.get('confidence', 0.0)
                if 'referent_clusters' in mapping:
                    entity.properties['clusters'] = list(mapping['referent_clusters'])
                if 'centroid' in mapping:
                    entity.embedding = mapping['centroid']
            count += 1
        return count

    def word_to_entity_id(self, word: str) -> str:
        return f'sym:{word}'

    def entity_to_words(self, entity_id: str) -> list:
        entity = self.graph.get_entity(entity_id)
        if entity and entity.type == 'symbol':
            return [entity.properties.get('text', entity_id.replace('sym:', ''))]
        return []
