# Database Guidelines

> INI-style text file persistence. No SQL, no ORM.

---

## Overview

Learner state is serialized to plain text files using INI-style key=value sections. No database server. The `ContinuousLearningLoop` adds a checkpoint manager on top.

---

## Persistence Mechanisms

### 1. Learner save/load (`learner_io.cpp`)

INI-style sectioned text file:

```
[meta]
version=1
stage=literacy
total_steps=1234

[config]
obs_dim=128
learning_rate=0.001

[knowledge_graph]
entities=42
relations=38

[statistics]
hippocampal_episodes=100
learning_progress=0.85
```

- `save(path)` writes via `std::ofstream`
- `load(path)` reads via `std::ifstream`
- Silent no-op if file cannot be opened (no throw)

### 2. Embedding model persistence (`embedding_trainer.cpp`)

Binary format via `std::ofstream` with `std::ios::binary`:

```cpp
std::ofstream out(path, std::ios::binary);
// Write: vocab_size (uint32), dim (uint32), then word + vector pairs
```

### 3. Checkpoint manager (`continuous_loop.cpp`)

JSON metadata alongside the learner state file:

```cpp
std::ofstream out(meta_path);
json meta = {{"checkpoint_id", id}, {"timestamp", ts}, {"reason", "auto"}};
out << meta.dump(2);
```

### 4. Server static files (`server.cpp`)

Reads web assets from disk via `std::ifstream` with binary mode for serving through Crow.

---

## File Layout Convention

```
data/
├── learner_state.txt        # Learner save/load
├── embeddings.bin           # Trained embedding vectors
├── checkpoints/
│   ├── ckpt_001.json        # Checkpoint metadata
│   └── ckpt_001_state.txt   # Learner state snapshot
```

---

## Common Mistakes

- **Don't assume JSON persistence** — core Learner uses INI text, not JSON
- **Don't forget path traversal check** in server routes: `if (path.find("..") != npos) reject`
- **Don't skip binary mode** for embedding files — `std::ios::binary` is required
- **Don't throw on file open failure** — silent no-op matches existing `learner_io.cpp` pattern
