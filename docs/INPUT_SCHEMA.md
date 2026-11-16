# CPS Input & Output Schema Specification

## 1. Input Task DAG (`dag.json`)

The `--dag <path>` file is a UTF-8 JSON object containing two top-level arrays: `nodes` and `edges`.

```json
{
  "nodes": [
    {
      "id": 0,
      "execution_time": 10.0,
      "CLB": 1200,
      "BRAM": 12,
      "DSP": 16,
      "EST": 0.0,
      "LFT": 25.0,
      "Slack": 15.0
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "communication": 40
    }
  ]
}
```

### Node Fields (`nodes[]`)
| Field | Type | Description |
|:------|:-----|:------------|
| `id` | `int` | Contiguous 0-based task identifier `0 .. N-1` (required by Stage-0 clustering). |
| `execution_time` | `number` | Non-negative task execution latency on the FPGA. |
| `CLB` | `int` | Required Configurable Logic Block resources ($\le 38850$). |
| `BRAM` | `int` | Required Block RAM resources ($\le 2100$). |
| `DSP` | `int` | Required DSP slice resources ($\le 2800$). |
| `EST` | `number` | Earliest Start Time of the task. |
| `LFT` | `number` | Latest Finish Time of the task (`LST` is derived as `LFT - execution_time`). |
| `Slack` | `number` | Static timing slack (`LFT - EFT` or `LST - EST`, $\ge 0$). |

### Edge Fields (`edges[]`)
| Field | Type | Description |
|:------|:-----|:------------|
| `source` | `int` | Producer task `id`. |
| `target` | `int` | Consumer task `id` (`target` cannot start until `source` completes). |
| `communication` | `int` | Data transfer volume weight ($\ge 0$) used in clustering affinity and Inter-Community Communication Cost (ICC). |

---

## 2. Optional Precomputed Communities (`--communities`)

When `--communities <path>` is omitted, `cps_cli` runs Stage-0 clustering automatically and saves `clustering_result.json` to `--output-dir`. You may pass an existing `clustering_result.json` via `--communities` to skip Stage 0. Required fields per entry in `communities[]`:
- `id` (`int`): Community identifier (1-based).
- `total_resources` (`{"CLB": int, "BRAM": int, "DSP": int}`): Aggregate hardware resource envelope of the community.
- `nodes` (`[{"id": int, ...}]`): Member task list partitioning all DAG tasks.

---

## 3. Output Files in `--output-dir`

### `schedule_result.json`
Primary machine-readable schedule artifact:
- `summary`:
  - `feasible` (`bool`): Whether a complete deadlock-free single-residency schedule was found.
  - `makespan` (`number | null`): Overall completion time $\max_v f(v)$.
  - `icc` (`number | null`): Total Inter-Community Communication Cost across all task edges.
  - `rtu` / `rtu_pct` (`number | null`): Resource-Time Utilization fraction and percentage (`weighted_task_work / resource_box_sum`).
  - `layout_valid` (`bool`): Result of checking all 9 Nested-DFX spatio-temporal invariants.
- `preset` & `preset_parameters`: Active scheduling preset configuration.
- `best_chromosome`: Evolved 20-dimensional policy vector (`alpha[7]`, `beta[11]`, `lambda`, `theta`).
- `communities[]`: Per-community activation window `[activation_time, completion_time)`, size class, assigned Parent/Slot IDs, `drr_rect`, `occupied_rect`, and community RTU%.
- `tasks[]`: Per-task start/finish timestamps and bound physical slot/version residence.
- `communication_edges`: Per-edge ICC relation (`same_slot_overlap`, `same_drr_reuse`, `same_subparent_final_full_cover_reuse`, `same_parent`, `diff_parent`), discrete hop coefficient $\in \{0, 1, 2, 3\}$, and weighted ICC cost.
- `nested_dfx_slots`: Explicit configuration tree, retained/full-cover slot table, composite reconfiguration groups, and `LOAD_GROUP` / `UNLOAD_GROUP` / configuration switch timeline.

### `schedule_report.md`
Human-readable Markdown summary table, community floorplan placement table, and chronological event trace.

### `convergence.csv`
Per-generation evolutionary trajectory (`generation,best_makespan,best_icc,best_rtu_pct,avg_feasible_makespan,feasible_population_count`).
