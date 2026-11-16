# CPS: Community-level Policy Scheduling for Nested-DFX FPGAs

**CPS** is a standalone C++17 scheduling and physical floorplanning framework for executing task-dependency DAGs on heterogeneous column-based FPGAs with **three-tier Nested Dynamic Function eXchange (Nested-DFX)** (`ParentRegion` $\to$ `SubParentRegion` $\to$ `ChildDRR`).

Instead of searching directly over combinatorial task-by-coordinate schedules, CPS decouples structural decomposition from evolutionary policy optimization across three stages:

1. **Stage 0 — Multi-Level Resource-Constrained Task Clustering**: Partitions the input task DAG into hardware-feasible task communities via multi-criteria edge affinity (communication volume, temporal window overlap, slack urgency), Louvain modularity coarsening, and overloaded-community post-splitting, accompanied by static quotient-graph cycle diagnostics.
2. **Stage 1 — Feasibility-Guided Policy Sampling**: Samples 20-dimensional real-valued community scheduling chromosomes $(\boldsymbol{\alpha}_{1..7}, \boldsymbol{\beta}_{1..11}, \lambda, \theta) \in [0, 1]^{20}$ through the event-driven single-residency decoder to seed a feasible initial population.
3. **Stage 2 — Genetic Policy Evolution & Nested-DFX Allocation**: Evolves policy chromosomes using Tournament Selection, Simulated Binary Crossover (SBX), Polynomial Mutation (PM), and stagnation-triggered diversity restarts, while the inner Nested-DFX allocator constructs and validates explicit slot/configuration hierarchies against 9 spatio-temporal invariants.

---

## Repository Structure

```text
CPS/
├── CMakeLists.txt                     # CMake build configuration (C++17, zero external deps)
├── include/cps/
│   ├── fpga_grid.hpp                  # Heterogeneous 146x350 column-based FPGA resource grid
│   ├── clustering.hpp                 # Stage-0 task clustering & quotient-SCC diagnostics
│   ├── preset_config.hpp              # Primary 'standard' mode & optional presets
│   ├── chromosome_io.hpp              # 20D policy chromosome representation & JSON I/O
│   ├── event_time.hpp                 # Continuous-time next-event clock advancement
│   ├── schedule_order.hpp             # Lexicographic schedule comparators
│   ├── slot_semantics.hpp             # Explicit nested slot/configuration reconciliation & 9 invariants
│   ├── nested_dfx_allocator.hpp       # 3-tier Nested-DFX candidate search, scoring & ICC model
│   ├── slot_export.hpp                # Reconfiguration group/switch event timeline exporter
│   ├── event_decoder.hpp              # Strict single-residency event-driven decoder
│   └── ga_engine.hpp                  # Stage-1 sampler, Stage-2 GA engine & report writers
├── src/
│   ├── clustering.cpp                 # Stage-0 clustering implementation
│   └── cps_main.cpp                   # Unified end-to-end CLI ('cps_cli')
├── tests/
│   └── slot_and_icc_tests.cpp         # Nested-DFX slot invariants & ICC regression suite
├── examples/                          # Reproducible benchmark DAGs (t30, t30_B, DenseNet121)
├── scripts/
│   └── run_examples.py                # Automated reproduction & verification script
├── docs/
│   └── INPUT_SCHEMA.md                # Input DAG & output JSON schema specification
└── third_party/nlohmann/json.hpp      # Bundled single-header JSON library
```

---

## Build & Unit Tests

### Prerequisites
- C++17 compiler (`g++ >= 9`, `clang++ >= 10`, or `MSVC >= 19.28`)
- CMake $\ge 3.16$
- Python $\ge 3.8$ *(optional, for `scripts/run_examples.py`)*

### Compile & Run CTest
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

---

## Quickstart Usage

Run the complete end-to-end CPS pipeline (Stage 0 $\to$ Stage 1 $\to$ Stage 2) on a task DAG under the default `standard` mode:

```bash
./build/cps_cli \
  --dag examples/t30/dag.json \
  --output-dir ./cps_out/t30 \
  --cluster-seed 202 \
  --seed 303
```

Each run writes four structured artifacts into `--output-dir`:
- `clustering_result.json` — Stage-0 community partition, per-community resource totals, and quotient-graph SCC diagnostics.
- `initial_solutions.json` — Stage-1 top feasible policy chromosomes.
- `convergence.csv` — Per-generation best Makespan, ICC, RTU%, and population feasibility trace.
- `schedule_result.json` & `schedule_report.md` — Final schedule metrics, task timelines, per-edge ICC breakdown, explicit Nested-DFX slot configurations, and event trace.

### Scheduling Presets (`--preset`)

CPS defaults to a single primary mode (`standard`) matching the baseline mathematical formulation, while exposing three optional multi-objective presets:

| Preset | CLI Flag | Outer Lexicographic Order | Inner Slot Scoring |
|:-------|:---------|:--------------------------|:-------------------|
| **Standard (Default)** | `--preset standard` *(or omit)* | Strict `(Makespan, ICC, -RTU)` | Pure structural/waste/defrag cost |
| **Comm-Aware** | `--preset comm-aware` | Strict `(Makespan, ICC, -RTU)` | + Incremental ICC guidance (`dmc_weight = 1.0`) |
| **Packing-Aware** | `--preset packing-aware` | `(Makespan, 3% ICC log-bucket, -RTU, ICC)` | + Area-time compactness (`reseff_weight = 1.0`) |
| **Balanced** | `--preset balanced` | `(Makespan, 1% ICC log-bucket, -RTU, ICC)` | Joint ICC + area-time compactness |

---

## Reproducing Included Examples

Three representative synthetic and real-world DNN workloads are included under `examples/` for immediate verification under the default `standard` mode and strict single-residency (`|L_k| = 1`) execution:

| Benchmark | Tasks / Edges | `--cluster-seed` | `--seed` | Makespan | ICC | RTU (%) |
|:----------|--------------:|-----------------:|---------:|---------:|----------:|--------:|
| `t30` | 32 / 47 | `202` | `303` | `730.0` | `742.0` | `27.37%` |
| `t30_B` | 32 / 81 | `42` | `404` | `545.0` | `1300.0` | `36.17%` |
| `DenseNet121` | 129 / 179 | `101` | `42` | `97553.0` | `2696640.0` | `27.04%` |

Run all three benchmarks and verify exact numerical agreement with a single command:

```bash
python scripts/run_examples.py
```

For detailed input/output JSON schemas, see [`docs/INPUT_SCHEMA.md`](docs/INPUT_SCHEMA.md).
