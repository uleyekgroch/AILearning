#include "ai_learning/domain/knowledge/knowledge_graph_cuda.cuh"

namespace ai_learning::domain::knowledge {

auto cuda_graph_upload(const CsrGraphData&) -> GpuGraphHandle* { return nullptr; }
void cuda_graph_release(GpuGraphHandle*&) {}
auto cuda_graph_handle_valid(const GpuGraphHandle*) -> bool { return false; }
auto cuda_graph_find_neighbors(const GpuGraphHandle*, const std::vector<NeighborQuery>&) -> BatchNeighborResult { return {}; }
auto cuda_graph_shortest_path(const GpuGraphHandle*, const std::vector<PathQuery>&) -> BatchPathResult { return {}; }
auto cuda_graph_entity_search(const GpuGraphHandle*, const std::vector<EntitySearchQuery>&) -> BatchEntitySearchResult { return {}; }

}  // namespace ai_learning::domain::knowledge