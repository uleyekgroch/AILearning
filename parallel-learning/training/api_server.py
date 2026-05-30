"""REST API服务器 — FastAPI

提供结构化的API接口：
- POST /api/learn — 学习新文本
- POST /api/query — 查询知识
- GET /api/stats — 获取统计
- POST /api/search — 搜索相似文档
- GET /api/health — 健康检查

运行方式：
    python training/api_server.py
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from training.integrated_ai import IntegratedAI
from training.gpu_optimized_learning import OptimizedLearningSystem
from training.knowledge_persistence import KnowledgePersistence


# ===== 数据模型 =====

class LearnRequest(BaseModel):
    """学习请求"""
    text: str = Field(..., description="要学习的文本")
    source: str = Field(default="api", description="来源标识")

class LearnBatchRequest(BaseModel):
    """批量学习请求"""
    texts: List[str] = Field(..., description="要学习的文本列表")
    sources: Optional[List[str]] = Field(default=None, description="来源标识列表")

class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="问题")
    top_k: int = Field(default=5, description="返回结果数量")

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=10, description="返回结果数量")


# ===== 响应模型 =====

class LearnResponse(BaseModel):
    """学习响应"""
    success: bool
    message: str
    stats: Dict

class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    confidence: float
    sources: List[Dict]
    stats: Dict

class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[Dict]
    total: int

class StatsResponse(BaseModel):
    """统计响应"""
    documents_processed: int
    triples_extracted: int
    entities_found: int
    concepts_formed: int
    causal_links: int
    analogies: int
    gpu_available: bool
    gpu_name: str


# ===== 全局状态 =====

app = FastAPI(
    title="AI学习系统API",
    description="基于8层架构的AI学习系统",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 系统实例
integrated_ai = None
gpu_system = None
persistence = None


def init_systems():
    """初始化系统"""
    global integrated_ai, gpu_system, persistence
    integrated_ai = IntegratedAI()
    gpu_system = OptimizedLearningSystem()
    persistence = KnowledgePersistence()
    print("系统初始化完成")


# ===== API端点 =====

@app.on_event("startup")
async def startup():
    """启动时初始化"""
    init_systems()


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": integrated_ai is not None,
    }


@app.post("/api/learn", response_model=LearnResponse)
async def learn(request: LearnRequest):
    """学习新文本"""
    try:
        # 使用集成AI系统学习
        result = integrated_ai.learn(request.text, source=request.source)

        # 同时使用GPU系统学习
        gpu_system.learn_batch([request.text], [request.source])

        return LearnResponse(
            success=True,
            message="学习成功",
            stats={
                "understanding_score": result.understanding_score,
                "concepts_formed": len(result.concepts_formed),
                "causal_links": len(result.causal_links),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learn/batch", response_model=LearnResponse)
async def learn_batch(request: LearnBatchRequest):
    """批量学习"""
    try:
        sources = request.sources or ["batch"] * len(request.texts)

        # 批量学习
        for text, source in zip(request.texts, sources):
            integrated_ai.learn(text, source=source)

        gpu_system.learn_batch(request.texts, sources)

        return LearnResponse(
            success=True,
            message=f"批量学习成功: {len(request.texts)} 篇",
            stats={
                "documents": len(request.texts),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """查询知识"""
    try:
        # 使用集成AI系统查询
        answer = integrated_ai.think(request.question)

        # 使用GPU系统搜索
        search_results = gpu_system.search(request.question, top_k=request.top_k)

        # 获取三元组
        query_result = gpu_system.query(request.question)

        return QueryResponse(
            answer=answer,
            confidence=0.8,
            sources=search_results,
            stats={
                "total_triples": query_result.get("total_triples", 0),
                "total_entities": query_result.get("total_entities", 0),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """搜索相似文档"""
    try:
        results = gpu_system.search(request.query, top_k=request.top_k)

        return SearchResponse(
            results=results,
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    """获取系统统计"""
    try:
        ai_stats = integrated_ai.get_stats()
        gpu_stats = gpu_system.get_stats()

        return StatsResponse(
            documents_processed=gpu_stats.get("documents_processed", 0),
            triples_extracted=gpu_stats.get("triples_extracted", 0),
            entities_found=gpu_stats.get("entities_found", 0),
            concepts_formed=ai_stats.get("abstraction", {}).get("total_concepts", 0),
            causal_links=ai_stats.get("causal", {}).get("total_links", 0),
            analogies=ai_stats.get("analogical", {}).get("total_analogies", 0),
            gpu_available=gpu_stats.get("gpu_available", False),
            gpu_name=gpu_stats.get("gpu_name", "N/A"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save")
async def save():
    """保存知识"""
    try:
        stats = integrated_ai.get_stats()
        filepath = persistence.save_knowledge(stats, "api_save")
        return {"success": True, "path": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/build-index")
async def build_index():
    """构建向量索引"""
    try:
        gpu_system.build_index()
        return {"success": True, "message": "索引构建完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== 主函数 =====

def main():
    print("=" * 70)
    print("AI学习系统 REST API")
    print("=" * 70)
    print("\n启动服务器...")
    print("API文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/api/health")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == '__main__':
    main()
