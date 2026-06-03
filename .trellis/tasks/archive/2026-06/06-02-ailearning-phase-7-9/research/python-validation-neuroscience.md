# Research: Python Validation / Neuroscience Module

- **Query**: Analyze src/validation/neuroscience.py for C++ port gap analysis
- **Scope**: internal
- **Date**: 2026-06-03

## Findings

### Module Overview

`src/validation/neuroscience.py` is a single file, **246 lines**. `__init__.py` is 1 line (docstring only).

### What It Does

This module validates the AI learning system's behavior against real infant neuroscience data. It compares system learning trajectories with published developmental benchmarks.

### Hardcoded Reference Data

The module contains four sets of real scientific data:

1. **INFANT_VOCAB_CDI** (lines 26-32) -- MacArthur-Bates CDI productive vocabulary norms. Maps age in months (8-72) to average vocabulary size (0-4000 words). Source: MacArthur-Bates Communicative Development Inventories.

2. **NEURAL_MARKERS** (lines 35-78) -- Six EEG neural marker timelines:
   - `MMN_acoustic` (0-6 months) -> `prediction_accuracy`
   - `MMN_phonemic` (6-12 months) -> `symbol_grounding_rate`
   - `N400_semantic` (12-24 months) -> `vocabulary_size`
   - `P600_syntactic` (18-36 months) -> `grammar_complexity`
   - `theta_word_learning` (10-20 months) -> `learning_progress`
   - `left_lateralization` (24-48 months) -> `composition_rate`
   
   Each marker maps to a system analogue metric.

3. **PIAGET_STAGES** (lines 81-86) -- Four Piaget cognitive development stages with age ranges and key abilities:
   - Sensorimotor (0-24 months): object permanence, goal-directed
   - Preoperational (24-72 months): symbolic play, language
   - Concrete operational (72-132 months): conservation, classification
   - Formal operational (132-192 months): abstract reasoning, hypothesis

4. **HABITUATION_DECAY** (lines 89-92) -- EEG habituation decay curve (trial 1-10, amplitude 1.0-0.33).

### Functions

#### `fit_curves(x_data, y_data) -> Dict` (lines 115-153)

Fits four curve models to data:
- Sigmoid: `L / (1 + exp(-k*(x-x0))) + b`
- Exponential: `a * exp(b*x) + c`
- Power law: `a * x^b + c`
- Linear: `a*x + b`

Uses `scipy.optimize.curve_fit` when available. Returns best model name, parameters, R-squared, and all model results. Falls back gracefully when scipy is unavailable.

#### `compute_correlation(system_data, infant_data) -> Dict` (lines 156-189)

Computes Pearson correlation between system metrics and infant benchmarks:
1. Finds common x-axis points (intersection of keys)
2. Normalizes both series to [0,1]
3. Computes Pearson r and p-value via `scipy.stats.pearsonr` (or manual calculation)
4. Returns `{correlation, p_value, n_points}`

#### `run_longitudinal(learner, num_rounds, sample_interval) -> Dict` (lines 196-246)

Main longitudinal comparison pipeline:
1. Runs `num_rounds` learning iterations (default 500)
2. Samples every `sample_interval` steps (default 50)
3. Maps step to month equivalent (step/num_rounds * 72 months)
4. Collects: vocabulary_size, prediction_accuracy, comm_success_rate, stage
5. Computes CDI vocabulary correlation
6. Fits growth curves
7. Checks neural marker emergence (value > 0.3 = "emerged")

Returns: `{snapshots, cdi_correlation, curve_fit, neural_markers, total_rounds}`

### Dependencies

- **scipy** (optional) -- `scipy.optimize.curve_fit`, `scipy.stats.pearsonr`
- **math** (stdlib)
- Requires a `learner` object with `get_stats()` method for `run_longitudinal()`

The `get_stats()` must return at minimum: `vocabulary_size`, `avg_error`, `comm_success_rate`, `stage`, `entity_count`.

### C++ Port Status

**No equivalent exists in C++.** There are no `validation*`, `neuroscience*`, or related files in `ai-learning-cpp/`.

### Suggested C++ Implementation Priority

**Priority: LOW** (value: MEDIUM, complexity: LOW)

Rationale:
- The module is self-contained (246 lines, no complex dependencies)
- The reference data is static and hardcoded
- Curve fitting requires either implementing from scratch or using a library (e.g., Ceres Solver, Eigen least-squares, or dlib)
- The `run_longitudinal()` function requires the C++ Learner to expose equivalent stats
- Useful for validation/evaluation but not critical for core learning functionality
- Could be implemented as a standalone test/benchmark tool rather than a core module

### Key Considerations for C++ Port

1. **Curve fitting**: scipy's `curve_fit` (Levenberg-Marquardt) has no direct C++ stdlib equivalent. Options:
   - Eigen's `NonLinearOptimization` module
   - Ceres Solver
   - Manual Gauss-Newton implementation
   - Simply port the linear/polynomial fits (skip sigmoid if not needed)

2. **Pearson correlation**: Straightforward to implement manually (already done in Python fallback).

3. **Learner interface**: Requires `get_stats()` returning the same keys. The C++ Learner already has `get_stats()` but needs to confirm the key names match.
