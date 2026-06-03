export const meta = {
  name: 'phase-7-remaining',
  description: 'Implement Phase 7.3 (Advanced API), 7.4 (WebSocket), 7.5 (Web Console) for AILearning REST service',
  phases: [
    { title: 'Parallel Build', detail: '7.3 Advanced API + 7.4 WebSocket in parallel' },
    { title: 'Web Console', detail: '7.5 Vue 3 + ECharts interactive dashboard' },
    { title: 'Verify', detail: 'Build + curl test all endpoints' },
  ],
}

// Phase 7.3 and 7.4 are independent
phase('Parallel Build')

const advancedAPI = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Implement Phase 7.3 - Advanced API Endpoints (Phase 3-6)\n\nPhase 7.1 done (crow + nlohmann/json), Phase 7.2 implementing core API.\nNow add Phase 3-6 advanced cognitive endpoints in server.cpp.\n\nEndpoints to implement:\n- POST /api/analogize -> analogical_transfer()\n- POST /api/protect -> protect_knowledge()\n- GET /api/forgetting -> detect_forgetting()\n- POST /api/abstract -> form_abstractions()\n- POST /api/observe-behavior -> observe_behavior()\n- POST /api/emotion -> process_emotion()\n- POST /api/insight -> try_insight()\n- POST /api/meta/recommend -> meta_recommend()\n- POST /api/meta/reflect -> meta_reflect()\n- POST /api/experiment/design -> design_experiment()\n- POST /api/experiment/record -> record_experiment()\n- POST /api/integrated/pipeline -> integrated_pipeline()\n- POST /api/integrated/meta-guided -> meta_guided_learn()\n- GET /api/integrated/emotion-params -> emotion_modulated_params()\n\nImplementation:\n1. Add routes in register_advanced_routes_() in server.cpp (accept crow::SimpleApp& if refactored)\n2. Each POST takes JSON body, returns JSON response\n3. Errors return {"error":"message"}, HTTP 400\n\nKey files to READ first:\n- ai-learning-cpp/src/server/server.cpp - current server implementation\n- ai-learning-cpp/src/server/server.hpp - server interface\n- ai-learning-cpp/include/ai_learning/core/learner.hpp - ALL Phase 3-6 method signatures\n- ai-learning-cpp/include/ai_learning/learning/integrated_learner.hpp - Phase 6 types\n- ai-learning-cpp/include/ai_learning/learning/meta_learner.hpp - Phase 5 types\n- ai-learning-cpp/include/ai_learning/learning/emotion_engine.hpp - EmotionState\n- ai-learning-cpp/include/ai_learning/learning/insight_engine.hpp - InsightEvent\n\nDo NOT modify Learner core code. Only modify server.cpp, server.hpp if needed.',
  { label: 'phase-7.3-advanced-api', phase: 'Parallel Build' }
)

const webSocket = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Implement Phase 7.4 - WebSocket Event Streaming\n\nAdd WebSocket support for real-time learning event push.\n\nImplementation:\n1. WebSocket endpoints in server.cpp:\n   - /ws/events - all learning events real-time push\n   - /ws/stats - stats summary push every second\n2. Event adapter - src/server/event_adapter.hpp/cpp:\n   - Convert internal events to JSON: {"type":"learn|reason|remember|stage_change|emotion|insight","timestamp":"...","data":{...}}\n   - Ring buffer of last 100 events, send buffer on new connection\n3. Heartbeat: ping every 30s, disconnect after 60s timeout\n\ncrow WebSocket usage:\n  CROW_WEBSOCKET_ROUTE(app, "/ws")\n    .onopen([&](crow::websocket::connection& conn){})\n    .onclose([&](crow::websocket::connection& conn, const std::string& reason){})\n    .onmessage([&](crow::websocket::connection& conn, const std::string& data, bool is_binary){})\n\nKey files:\n- ai-learning-cpp/src/server/server.cpp - add WS routes\n- ai-learning-cpp/src/server/server.hpp - add WS management\n- ai-learning-cpp/include/ai_learning/domain/domain_events.hpp - IEventPublisher\n- ai-learning-cpp/CMakeLists.txt - add event_adapter.cpp to sources\n\nNew files: src/server/event_adapter.hpp, src/server/event_adapter.cpp\nDo NOT modify Learner core code.',
  { label: 'phase-7.4-websocket', phase: 'Parallel Build' }
)

log('Phase 7.3 result: ' + (advancedAPI ? 'done' : 'failed'))
log('Phase 7.4 result: ' + (webSocket ? 'done' : 'failed'))

// Phase 7.5: Web Console (depends on 7.3 + 7.4)
phase('Web Console')

const webConsole = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Implement Phase 7.5 - Web Interactive Console\n\nCreate frontend web console at ai-learning-cpp/web/\n\nFiles to create:\n1. web/index.html - main page with Vue 3 + ECharts CDN\n2. web/app.js - Vue 3 app logic\n3. web/style.css - styling\n\nFeatures:\n- Dashboard: learning progress chart (ECharts line), knowledge count, stage\n- Emotion panel: valence/arousal/dominance radar chart\n- Knowledge graph: force-directed graph (ECharts graph)\n- Operations: text input to learn, Q&A dialog, trigger reasoning/insight/analogize\n- System status: development stage, statistics\n\nTech stack (all CDN, no build step):\n- Vue 3: https://unpkg.com/vue@3/dist/vue.global.js\n- ECharts: https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\n- Pure HTML/CSS/JS\n\nAlso update server.cpp to serve static files:\n  app.static_dir("web/"); // or manual route\n\nLayout: left-right split, dashboard+graph on left, operations+results on right.\n\nKey files:\n- ai-learning-cpp/src/server/server.cpp - add static file serving\n- ai-learning-cpp/src/server/server_config.hpp - static_dir config\n\nDo NOT modify any C++ core files. Only add web/ files and update server static serving.',
  { label: 'phase-7.5-web-console', phase: 'Web Console' }
)

log('Phase 7.5 result: ' + (webConsole ? 'done' : 'failed'))

// Phase: Verify build
phase('Verify')

const verifyResult = await agent(
  'Active task: .trellis/tasks/06-02-ailearning-phase-7-9\n\n## Verify Phase 7 Full Build\n\nSteps:\n1. cd ai-learning-cpp/build-server\n2. Run: cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release\n3. Run: cmake --build . --target ai_learning_server -j8\n4. If build fails, read errors and FIX them directly\n5. If build succeeds, test with:\n   - PATH="/d/msys64/ucrt64/bin:$PATH" ./ai_learning_server.exe --port 8765 &\n   - sleep 3\n   - curl -s http://127.0.0.1:8765/api/health\n   - curl -s -X POST http://127.0.0.1:8765/api/learn/text -H "Content-Type: application/json" -d \'{"text":"hello world","source":"test"}\'\n   - curl -s http://127.0.0.1:8765/api/stats\n   - kill server\n6. Report results\n\nImportant: Windows needs PATH="/d/msys64/ucrt64/bin:$PATH" to find DLLs.\nIf CMakeLists.txt needs updates for new source files (event_adapter.cpp etc), update it.',
  { label: 'verify-build', phase: 'Verify' }
)

log('Verify result: ' + (verifyResult ? 'done' : 'failed'))

return {
  advancedAPI: advancedAPI ? 'completed' : 'failed',
  webSocket: webSocket ? 'completed' : 'failed',
  webConsole: webConsole ? 'completed' : 'failed',
  verify: verifyResult ? 'completed' : 'failed',
}
