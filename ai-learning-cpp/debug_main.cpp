#include <iostream>
#include "ai_learning/domain/knowledge/knowledge_graph.hpp"
#include "ai_learning/domain/knowledge/entity.hpp"
#include "ai_learning/domain/knowledge/relation.hpp"
#include "ai_learning/learning/knowledge_extractor.hpp"

auto main() -> int {
    using namespace ai_learning::domain::knowledge;

    KnowledgeGraph kg;
    kg.add_entity(Entity("人工智能", "concept"));
    kg.add_entity(Entity("计算机科学", "concept"));
    kg.add_relation(Relation("人工智能", "计算机科学", "是", 0.9));
    std::cout << "KG: entities=" << kg.entity_count() << " relations=" << kg.relation_count() << "\n" << std::flush;

    auto rels = kg.get_relations_of("人工智能");
    std::cout << "Relations: " << rels.size() << "\n" << std::flush;
    for (const auto& r : rels) {
        std::cout << "  " << r.get().source_id() << " -[" << r.get().type() << "]-> " << r.get().target_id() << "\n" << std::flush;
    }

    // Now test with KnowledgeExtractor
    std::cout << "\nTesting entity extraction\n" << std::flush;
    auto entities = ai_learning::learning::KnowledgeExtractor::extract_entities(
        "人工智能是计算机科学的一个分支");
    std::cout << "Found " << entities.size() << " entities\n" << std::flush;

    std::cout << "\nTesting triple extraction\n" << std::flush;
    auto triples = ai_learning::learning::KnowledgeExtractor::extract_triples(
        "人工智能是计算机科学的一个分支", entities);
    std::cout << "Found " << triples.size() << " triples\n" << std::flush;
    for (const auto& t : triples) {
        std::cout << "  " << t.subject << " -[" << t.relation << "]-> " << t.object << "\n" << std::flush;
    }

    std::cout << "\nDone!\n" << std::flush;
    return 0;
}
