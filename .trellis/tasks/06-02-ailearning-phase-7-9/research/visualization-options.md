# Research: Real-time Visualization for AILearning System

- **Query**: Options for real-time visualization of learning metrics from the C++ cognitive learning system
- **Scope**: External (technology options and patterns)
- **Date**: 2026-06-02

## Findings

### Project Context

The AILearning C++ system generates real-time learning metrics during operation:

**Data sources from Learner class**:
- `get_stats()` — returns `std::map<std::string, double>` (error rates, learning progress, step counts)
- `learning_progress()` — returns `ProgressSnapshot` (mastery, coverage, efficiency)
- `emotion_engine().current_state()` — EmotionState (valence, arousal, dominance, label)
- `metacognitive_report()` — MetacognitiveReport (knowledge gaps, confidence levels)
- `skill_tree()` — skill nodes and mastery levels
- `knowledge_graph()` — entity/relation counts, type distribution
- `engine().error_history()` — prediction error deque over time
- `check_milestones()` — development milestone events

**Operations producing streams of data**:
- `autonomous_learn()` — runs 1000+ steps, produces step-by-step metrics
- `integrated_pipeline()` — six-step pipeline with per-step reports
- `autonomous_learning_run()` — multi-iteration loop with progress
- `experiment_driven_explore()` — exploration with hypothesis updates

### Visualization Approaches

#### Approach 1: WebSocket + Web Dashboard (Recommended)

**Architecture**: C++ server pushes metrics via WebSocket to a browser-based dashboard.

```
C++ Backend (crow/cpp-httplib)
    |
    | WebSocket /ws/v1/metrics
    |
    v
Browser (React/Vue + Chart library)
    - Real-time charts (error history, learning curves)
    - Knowledge graph visualization (D3.js/Cytoscape)
    - Emotion state display (valence-arousal plane)
    - Skill tree progress bars
    - Milestone timeline
```

**Components**:

| Component | Options | Notes |
|-----------|---------|-------|
| WebSocket server (C++) | websocketpp, libwebsockets, oat++ WebSocket | websocketpp is header-only, Boost.Asio based |
| Frontend framework | React, Vue 3, Svelte | Vue 3 is simplest for a single dashboard |
| Chart library | Chart.js, ECharts, Plotly.js, Recharts | ECharts best for real-time streaming |
| Graph visualization | D3.js, Cytoscape.js, vis-network | Cytoscape.js best for knowledge graphs |
| Build tool | Vite | Fast HMR, simple setup |

**Data streaming pattern**:

```cpp
// C++ side: push metrics during learning loop
class MetricsBroadcaster {
public:
    void broadcast(const std::string& event_type, const nlohmann::json& data) {
        std::string msg = json{{"type", event_type}, "data", data}}.dump();
        for (auto& ws : connections_) {
            ws->send(msg);
        }
    }

    void on_learning_step(int step, double error, double progress) {
        broadcast("learning_step", {
            {"step", step},
            {"error", error},
            {"progress", progress},
            {"timestamp", /* ms */ }
        });
    }

    void on_emotion_update(const EmotionState& state) {
        broadcast("emotion", {
            {"valence", state.valence},
            {"arousal", state.arousal},
            {"label", state.label}
        });
    }
};
```

```javascript
// Frontend side: receive and render
const ws = new WebSocket('ws://localhost:8080/ws/v1/metrics');
ws.onmessage = (event) => {
    const { type, data } = JSON.parse(event.data);
    switch(type) {
        case 'learning_step':
            errorChart.appendData([data.step, data.error]);
            progressChart.appendData([data.step, data.progress]);
            break;
        case 'emotion':
            emotionPlane.update(data.valence, data.arousal);
            break;
    }
};
```

**Pros**:
- Real-time with low latency (~ms)
- Rich visualization possibilities (any JS library)
- Cross-platform (browser)
- Can serve the dashboard from the same C++ server
- Remote monitoring possible

**Cons**:
- More development effort (frontend + backend)
- WebSocket library needed in C++ (websocketpp or libwebsockets)
- Browser resource usage for heavy graph rendering

#### Approach 2: Python Wrapper + Plotly Dash / Streamlit

**Architecture**: Use pybind11 bindings to expose C++ Learner to Python, then use Python visualization frameworks.

```python
# Using pybind11 bindings
import ai_learning as al

learner = al.Learner(al.LearnerConfig())
result = learner.learn_from_text("Physics is the study of matter and energy")

# Plotly Dash dashboard
import dash
from dash import dcc, html
import plotly.graph_objects as go

app = dash.Dash(__name__)

@app.callback(Output('error-chart', 'figure'),
              Input('interval', 'n_intervals'))
def update_chart(n):
    stats = learner.get_stats()
    # Update charts
    return fig
```

**Pros**:
- Fastest to prototype (Python ecosystem)
- Plotly Dash provides real-time WebSocket updates
- Rich scientific visualization (Plotly, Matplotlib)
- Jupyter notebook integration for interactive exploration
- Leverages the pybind11 bindings being built anyway

**Cons**:
- Additional Python dependency layer
- Performance overhead for high-frequency metrics (Python GIL)
- Less suitable for production deployment
- Real-time streaming from C++ requires careful GIL management

#### Approach 3: Dear ImGui (Native Desktop)

**Architecture**: Embed Dear ImGui directly in the C++ application for an immediate-mode GUI.

```cpp
#include "imgui.h"
#include "imgui_impl_glfw.h"

void render_dashboard(Learner& learner) {
    ImGui::Begin("AILearning Dashboard");

    // Learning progress
    auto stats = learner.get_stats();
    ImGui::Text("Total Steps: %d", (int)stats["total_steps"]);
    ImGui::ProgressBar(stats["learning_progress"]);

    // Error history chart (ImPlot)
    auto& errors = learner.engine().error_history();
    ImPlot::PlotLine("Prediction Error", errors.data(), errors.size());

    // Emotion state
    auto& emotion = learner.emotion_engine().current_state();
    ImGui::Text("Emotion: %s", emotion.label.c_str());
    ImGui::SliderFloat("Valence", &emotion.valence, -1, 1);

    // Knowledge graph stats
    auto& kg = learner.knowledge_graph();
    ImGui::Text("Entities: %d  Relations: %d",
                kg.entity_count(), kg.relation_count());

    ImGui::End();
}
```

**Pros**:
- Zero external dependencies beyond GLFW/SDL2
- Immediate mode — no frontend/backend split
- Very low latency (same process)
- Integrated with the C++ build
- Great for development/debugging

**Cons**:
- Not suitable for remote monitoring
- Limited chart types compared to web libraries
- No knowledge graph visualization (would need custom rendering)
- Desktop-only (no mobile/browser access)
- Dear ImGui is primarily a tool for developers, not end users

#### Approach 4: Matplotlib (via pybind11)

**Architecture**: Use pybind11 to call into C++ Learner from Python scripts, visualize with Matplotlib.

```python
import matplotlib.pyplot as plt
import ai_learning as al

learner = al.Learner(al.LearnerConfig())

# Run learning and collect data
errors = []
for step in range(1000):
    result = learner.learn_from_experience(obs, action, next_obs, reward)
    stats = learner.get_stats()
    errors.append(stats["prediction_error"])

plt.plot(errors)
plt.xlabel("Step")
plt.ylabel("Prediction Error")
plt.title("Learning Curve")
plt.show()
```

**Pros**:
- Simplest for scientific visualization
- No server infrastructure needed
- Familiar to ML researchers
- Publication-quality figures

**Cons**:
- Not real-time (batch visualization)
- No interactive dashboard
- No streaming capability

#### Approach 5: Plotly Dash with Real-time Streaming

**Architecture**: Combine pybind11 bindings with Dash's WebSocket-based live updates.

```python
# dash_app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import ai_learning as al
import threading

learner = al.Learner(al.LearnerConfig())
app = dash.Dash(__name__)

# Layout with real-time charts
app.layout = html.Div([
    dcc.Graph(id='error-chart'),
    dcc.Graph(id='knowledge-graph'),
    dcc.Graph(id='emotion-plane'),
    dcc.Interval(id='update', interval=500),  # 500ms refresh
])

@app.callback(
    Output('error-chart', 'figure'),
    Input('update', 'n_intervals'))
def update_error_chart(n):
    stats = learner.get_stats()
    # Build figure from stats
    fig = go.Figure(go.Scatter(y=error_history))
    return fig
```

**Pros**:
- Real-time updates with Dash callbacks
- Professional-looking dashboard
- Interactive charts (zoom, pan, hover)
- Python ecosystem flexibility

**Cons**:
- Python GIL limits streaming frequency
- Requires running both C++ Learner and Python Dash server
- Polling-based (500ms interval) rather than true push

### WebSocket Library Options for C++

| Library | License | Header-only | Windows | C++20 | Notes |
|---------|---------|-------------|---------|-------|-------|
| websocketpp | BSD | Yes | Yes | Yes | Boost.Asio based, mature |
| libwebsockets | MIT | No | Yes | Yes | C library, very performant |
| oat++ WebSocket | Apache 2.0 | No | Partial | Yes | Integrated with oat++ framework |
| uWebSockets | Apache 2.0 | No | Yes | Yes | Extremely fast, used by Socket.IO |
| ixwebsocket | BSD-3 | Yes | Yes | Yes | Lightweight, TLS support |

**Recommended for this project**: `uWebSockets` (best performance) or `websocketpp` (simplest integration with Boost.Asio).

### Data Streaming Patterns

#### Pattern 1: Server-Push (Recommended)

The C++ server pushes metrics as they are generated. Best for real-time visualization during learning.

```
C++ Learner (learning loop)
    |
    v [each step]
MetricsBroadcaster
    |
    v [WebSocket frame]
Browser/Client
    |
    v
Chart update
```

#### Pattern 2: Polling

Client requests current state at intervals. Simpler but higher latency.

```
GET /api/v1/learner/stats → {error: 0.02, progress: 0.85, ...}
GET /api/v1/learner/emotion → {valence: 0.3, arousal: 0.7, ...}
```

Polling is simpler to implement (no WebSocket needed) but wastes resources when nothing changes and has inherent latency.

#### Pattern 3: Event-Sourced

Store all events in a log, clients subscribe to event stream. Best for replay/debugging.

```
C++ → emit DomainEvent (already exists in codebase via IEventPublisher)
    → EventStore (append-only log)
    → WebSocket subscribers receive filtered events
```

**The existing `IEventPublisher` interface** (`domain_events.hpp`) already provides the event infrastructure. A WebSocket publisher implementation could be added:

```cpp
class WebSocketEventPublisher : public domain::IEventPublisher {
public:
    void publish(domain::DomainEvent event) override {
        nlohmann::json j = std::visit([](auto& e) -> nlohmann::json {
            // Convert each event type to JSON
            return serialize_event(e);
        }, event);
        broadcaster_.broadcast("domain_event", j);
    }
};
```

### Recommended Visualization Strategy

**For development/debugging**: Dear ImGui embedded in the C++ app. Fast iteration, no external dependencies.

**For research presentation**: Python + Matplotlib via pybind11. Publication-quality figures, familiar to ML community.

**For real-time monitoring dashboard** (Phase 7-9 target): WebSocket + Vue 3 + ECharts.

**Recommended stack**:

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| WebSocket server | uWebSockets or websocketpp | C++ side, push metrics |
| HTTP REST server | crow | API endpoints, serve static files |
| Frontend | Vue 3 + Vite | Simple, fast, single-page dashboard |
| Charts | Apache ECharts | Best real-time streaming charts |
| Knowledge graph | Cytoscape.js | Interactive graph visualization |
| JSON serialization | nlohmann/json | Already recommended for REST API |

### Dashboard Layout (Conceptual)

```
+-----------------------------------------------+
| AILearning Dashboard                     [v0.2]|
+--------+--------------------------------------+
| Sidebar |  Main Content Area                  |
|         |                                      |
| [Stats] |  +------------------+ +-----------+ |
| [KG]    |  | Learning Curve   | | Emotion   | |
| [Emotion|  | (ECharts line)   | | (VA plane)| |
| [Skills]|  +------------------+ +-----------+ |
| [Miles.]|                                      |
|         |  +------------------+ +-----------+ |
|         |  | Knowledge Graph  | | Skill Tree| |
|         |  | (Cytoscape.js)   | | (progress)| |
|         |  +------------------+ +-----------+ |
|         |                                      |
| [Learn] |  +----------------------------------+|
| [Think] |  | Milestone Timeline               ||
| [Reason]|  | (horizontal scroll)              ||
|         |  +----------------------------------+|
+--------+--------------------------------------+
```

### External References

- [Apache ECharts](https://echarts.apache.org/) — best-in-class real-time chart library, supports large datasets
- [Cytoscape.js](https://js.cytoscape.org/) — graph theory visualization, good for knowledge graphs
- [Vue 3](https://vuejs.org/) — progressive JavaScript framework
- [Dear ImGui](https://github.com/ocornut/imgui) — immediate-mode GUI for C++
- [Plotly Dash](https://dash.plotly.com/) — Python framework for ML dashboards
- [websocketpp](https://github.com/zaphoyd/websocketpp) — C++ WebSocket library
- [uWebSockets](https://github.com/uNetworking/uWebSockets) — high-performance C++ WebSocket
- [ImPlot](https://github.com/epezent/implot) — Dear ImGui plotting extension

### Related Specs

- `ai-learning-cpp/include/ai_learning/domain/domain_events.hpp` — existing event system (IEventPublisher)
- `ai-learning-cpp/include/ai_learning/learning/emotion_engine.hpp` — emotion metrics for visualization
- `ai-learning-cpp/include/ai_learning/learning/skill_tree.hpp` — skill tree data for progress display
- `ai-learning-cpp/include/ai_learning/domain/knowledge/knowledge_graph.hpp` — KG data for graph visualization

## Caveats / Not Found

- **No existing visualization code**: The project has no UI or visualization components.
- **Event streaming integration**: The `IEventPublisher` interface exists but is optional (publisher can be nullptr). For visualization, a WebSocket publisher implementation would need to be created.
- **Performance of graph rendering**: Large knowledge graphs (5000+ entities) may need D3.js force layout with WebGL or Cytoscape.js with layout optimization.
- **Cross-platform rendering**: Dear ImGui needs OpenGL/Vulkan/Metal backend. On Windows, GLFW + OpenGL is simplest.
- **Real-time frequency**: Learning steps may produce data at high frequency (1000+ Hz). Dashboard should aggregate/downsample for display (e.g., show every 10th step).
