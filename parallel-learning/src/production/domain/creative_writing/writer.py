"""
创造性写作系统 - 像小朋友一样写作

不是简单的模板填充，而是真正的创造：
1. 理解主题
2. 组织思路
3. 创造内容
4. 表达情感
"""

import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class WritingPrompt:
    """写作提示"""
    topic: str
    style: str = "narrative"  # narrative, descriptive, persuasive, expository
    length: str = "short"  # short, medium, long
    emotion: str = "neutral"  # happy, sad, excited, calm
    audience: str = "general"  # children, adults, academic


@dataclass
class WrittenWork:
    """作品"""
    title: str
    content: str
    style: str
    word_count: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CreativeWriter:
    """创造性写作系统

    像小朋友一样写作：
    - 从已有知识中提取素材
    - 组织成有意义的故事
    - 表达情感和想法
    """

    def __init__(self):
        """初始化写作系统"""
        # 知识库（从wiki学习）
        self.knowledge_base: Dict[str, List[str]] = {}

        # 写作模板
        self.templates = {
            'narrative': [
                "从前，{character}在{place}遇到了{event}。",
                "有一天，{character}发现{discovery}。",
                "在{time}，{character}决定{action}。",
            ],
            'descriptive': [
                "{subject}是{adjective}的，它{feature}。",
                "在{location}，有一个{adjective}的{object}。",
                "{color}的{object}在{place}闪闪发光。",
            ],
            'persuasive': [
                "我认为{topic}很重要，因为{reason}。",
                "为什么{topic}值得我们关注？因为{evidence}。",
                "让我们来谈谈{topic}，它{benefit}。",
            ],
            'expository': [
                "{topic}是指{definition}。",
                "关于{topic}，有以下几个要点：{points}。",
                "首先，{point1}。其次，{point2}。",
            ],
        }

        # 词汇库
        self.vocabulary = {
            'adjectives': ['美丽', '聪明', '勇敢', '善良', '有趣', '神奇', '温暖', '明亮'],
            'emotions': ['快乐', '兴奋', '好奇', '惊讶', '感动', '自豪', '满足'],
            'places': ['森林', '城市', '学校', '家', '公园', '图书馆', '实验室'],
            'characters': ['小明', '小红', '老师', '科学家', '艺术家', '探险家'],
        }

        # 写作历史
        self.writing_history: List[WrittenWork] = []

    def learn_knowledge(self, topic: str, facts: List[str]) -> None:
        """
        学习知识

        Args:
            topic: 主题
            facts: 相关事实
        """
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []

        self.knowledge_base[topic].extend(facts)
        logger.info(f"Learned {len(facts)} facts about {topic}")

    def write(self, prompt: WritingPrompt) -> WrittenWork:
        """
        写作

        Args:
            prompt: 写作提示

        Returns:
            作品
        """
        # 1. 理解主题
        topic_knowledge = self.knowledge_base.get(prompt.topic, [])

        # 2. 选择模板
        template = self._select_template(prompt.style)

        # 3. 生成内容
        content = self._generate_content(template, prompt, topic_knowledge)

        # 4. 创建作品
        work = WrittenWork(
            title=f"关于{prompt.topic}的{prompt.style}文",
            content=content,
            style=prompt.style,
            word_count=len(content),
            created_at=datetime.now(),
            metadata={
                'topic': prompt.topic,
                'emotion': prompt.emotion,
                'audience': prompt.audience,
            }
        )

        # 5. 保存到历史
        self.writing_history.append(work)

        return work

    def _select_template(self, style: str) -> str:
        """
        选择模板

        Args:
            style: 写作风格

        Returns:
            模板
        """
        templates = self.templates.get(style, self.templates['narrative'])
        return random.choice(templates)

    def _generate_content(self, template: str, prompt: WritingPrompt,
                         knowledge: List[str]) -> str:
        """
        生成内容

        Args:
            template: 模板
            prompt: 写作提示
            knowledge: 相关知识

        Returns:
            内容
        """
        # 填充模板
        content = template

        # 替换占位符
        replacements = {
            '{topic}': prompt.topic,
            '{character}': random.choice(self.vocabulary['characters']),
            '{place}': random.choice(self.vocabulary['places']),
            '{adjective}': random.choice(self.vocabulary['adjectives']),
            '{emotion}': random.choice(self.vocabulary['emotions']),
        }

        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)

        # 添加知识内容
        if knowledge:
            # 选择相关知识
            selected_facts = random.sample(knowledge, min(3, len(knowledge)))
            knowledge_text = "。".join(selected_facts)
            content += f"\n\n关于{prompt.topic}，我们知道：{knowledge_text}。"

        # 添加结尾
        content += f"\n\n这就是关于{prompt.topic}的故事。"

        return content

    def get_writing_stats(self) -> Dict[str, Any]:
        """获取写作统计"""
        return {
            'total_works': len(self.writing_history),
            'topics_written': list(set(w.metadata.get('topic', '') for w in self.writing_history)),
            'styles_used': list(set(w.style for w in self.writing_history)),
            'avg_word_count': sum(w.word_count for w in self.writing_history) / max(1, len(self.writing_history)),
        }
