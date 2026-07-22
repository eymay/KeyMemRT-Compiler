# CGO 2027 Paper #108 — Rebuttal Preparation Material

This document consolidates the reviewers' criticisms into unified points and
collects the implementation facts from this repository that answer them.
Reviewer tags: A (expert, major revision), B (knowledgeable, major revision),
C (expert, reject), D (outsider, major revision), E (outsider, reject).

---

## Part 1 — Consolidated criticism points

### C1. No ablation study isolating the key-lifetime-management contribution (A, C, D, E)
The pipeline mixes the proposed rotation-key lifetime management with BSGS
decomposition, rotation hoisting, bootstrap removal, and ciphertext/plaintext
clearing. Reviewers cannot tell how much of the reported gain comes from the
core idea. Specific sub-questions:
- Why does Low-Memory mode beat the all-RAM ANT-ACE baseline on ResNet-50,
  LoLa, LeNet even though all key loading is delayed to execution time? (A)
- How much comes from BSGS, hoisting, bootstrap removal, clear-ops
  individually? (C, D)
- Ablation of live-range merging itself (E).

### C2. Balanced-mode comparison may be unfair due to the extra prefetch thread (A)
KeyMemRT's background prefetching thread gives it a parallelism advantage the
baselines lack. Requested: either give the baseline the same extra core (e.g.
OpenMP-accelerated OpenFHE) or ablate with prefetching disabled but liveness
analysis kept.

### C3. Key movement / deserialization overhead is never quantified (A, C, D, E)
- Absolute cost of bringing one ~130 MB rotation key from disk into memory,
  split into disk I/O vs deserialization (A, C).
- How much of that cost prefetching actually hides, and under what
  conditions (C).
- Storage medium, I/O bandwidth, and OS page-cache effects: the machine has
  512 GB RAM vs ≤58 GB of keys, so the OS block cache probably hides most I/O
  while consuming memory not counted in the reported footprint (E). This is
  the sharpest evaluation criticism and needs a direct answer (e.g. runs with
  `cgroup` memory limits and/or dropped page caches).

### C4. Missing algorithmic detail — merge policy, prefetch policy, runtime policy (A, C, D, E)
- What "close in distance" means in the Merge Rotation Keys pass: metric,
  threshold, how selected (C, D, E — asked verbatim by three reviewers).
- How loops, branches, and repeated distant uses of the same key are
  handled (C, D).
- How the compiler chooses prefetch insertion points/distances so that I/O
  latency is hidden (A).
- Runtime policy: prefetch queue scheduling under a memory limit,
  prioritization when memory is saturated, behavior of the user-provided
  prefetched-key limit in Balanced mode (A, C, E).

### C5. Sensitivity to execution order / scheduling (C)
Key lifetimes are optimized for one fixed execution order; different valid
schedules of independent dataflow ops could change lifetime overlap, peak
memory, and reload counts. Scheduling itself is not part of the optimization,
and worst cases (many overlapping lifetimes, repeated reloads) are not
characterized.

### C6. Memory-budget treatment is underdeveloped (C, D)
- Why is runtime information needed at all — could a memory-budget-aware
  compiler schedule key movement statically? (C)
- Only two operating modes are shown; the continuous memory–latency
  trade-off under different budgets is unexplored (C).
- No experiment on a real system under tight memory limits demonstrating that
  KeyMemRT runs programs that otherwise fail (D).

### C7. Bootstrap-removal correctness (C, D)
Why is bootstrap removal part of this work, how is its correctness ensured,
and what is its individual impact on runtime? Numerical-accuracy validation
of outputs is requested (D).

### C8. Framing / overclaiming of "unlocking" scalability (B)
- The memory reduction is already reachable by other systems (Fhelipe points
  would sit near KeyMemRT's in Fig. 1); the real contribution is the same
  memory at lower runtime cost.
- Non-RAM costs of large key material (generation, network transfer, disk
  storage) are untouched.
- The linear trend in Fig. 1 cannot continue: rotation-index count saturates
  at 2·sqrt(N) (= 512 here) under BSGS, so the extrapolation is misleading.
- Parameter selection and rotation chaining already give a rich trade-off
  space; the paper should position itself inside it, not as a unique unlock.

### C9. Evaluation breadth and reporting detail (E, D)
- All benchmarks are NNs; add non-NN workloads.
- Add Orion as a baseline (it already does hand-crafted coarse-grained key
  loading from disk).
- Report compile-time/pass overhead on the largest programs (ResNet-50).
- Report number of inserted load/clear ops, whether prefetch hides all
  latency, memory contention, hardware/storage details.

### C10. Writing and structure (E)
No clear storyline from motivation → gap → background → implementation;
motivating example precedes FHE background; Sections 2 and 4 feel
disconnected and repetitive.

---

## Part 2 — Implementation facts from this repository

These are the ground-truth answers the codebase supports, with citations.

### 2.1 The merge pass: "close in distance" (answers C4 for reviewers C, D, E)

The operative analysis is `RotationKeyLivenessDFA`
(`lib/Dialect/KMRT/Transforms/RotationKeyLivenessDFA.cpp`), a dataflow
analysis over key states {Loaded, Cleared, NotLoaded, Unknown}, run together
with dead-code analysis and constant propagation
(`MergeRotationKeys.cpp:98-108`).

Initial insertion: every `openfhe.rot` is converted to
`load_key` → `rotation` → `clear_key`
(`lib/Dialect/KMRT/Conversions/OpenfheToKMRT/OpenfheToKMRT.cpp:32-63`);
bootstrap keys get per-bootstrap load/clear from `bootstrap-rotation-analysis`.

Distance metric and thresholds (`lib/Dialect/KMRT/Transforms/MergeRotationKeys.cpp`):
- **Sequential clear→load merge**: forward scan from a `clear_key` for a
  `load_key` of the same index; distance = number of intervening *FHE ops*
  (dialect `openfhe`; key-management ops are free), with loops weighted by
  trip count (`countFHEOps`, `:221-258`). Threshold: `maxDistance = 10`
  (`:278-279`).
- **Preloop merge**: forward window of 20 ops to find an `affine.for`
  (`:144`), then an affine-set analysis (`AffineSetAnalysis.cpp:231-304`)
  proves the preloaded key coincides with a loop iteration (direct IV or
  giant-step `iv*step` forms); the loop is rewritten with `affine.if (iv==K)`
  guards to skip the redundant in-loop load/clear (`:999-1149`).
- **Postloop merges**: backward windows of 50 and 100 ops (`:433,:540,:644`).

Honest internal caveats (decide whether to concede in rebuttal or fix first):
- Thresholds are hard-coded constants, not pass options — the `.td` prose
  says "configurable (default 10)" (`MergeRotationKeys.td:44-46`) but the
  only actual option is `enable-loop-peeling` (`:56-59`). Making the
  thresholds pass options is a trivial change and a good revision item.
- No cost model; pairing of clear↔load is by linear textual proximity,
  sanity-gated by the DFA state (`:127-131`).

### 2.2 Loops, branches, repeated uses (answers C4/C5)

- Loops are handled **structurally on rolled `affine.for`**, not by
  unrolling: trip-count-weighted distance, affine-set intersection for
  IV-dependent key indices, `affine.if` guards, extension of BSGS-emitted
  guards (`MergeRotationKeys.cpp:801-897`), and loop peeling that hoists
  inner-loop key management into pre/post loops (`:1197-1343`).
- Branches: the DFA joins conservatively to `Unknown` at control-flow merges
  (`RotationKeyLivenessDFA.cpp:25-39`), which suppresses merging; the
  rewrite itself only recognizes `affine.for`/`affine.if` shapes. For the
  evaluated ML workloads the post-lowering IR is straight-line + affine
  loops, so this is sufficient — but it is a real scope limitation to state.
- Repeated distant uses beyond the windows are reloaded — this is by design
  the memory-side of the trade-off (the alternative is keeping the key
  resident, i.e. drifting back toward ANT-ACE behavior).
- **Fixed execution order (reviewer C is right)**: all scans are
  linear next/prev-node walks in block order; scheduling is not co-optimized.
  Rebuttal angle: KeyMemRT runs as a *post-optimization* stage on the order
  produced by the upstream compiler, so any scheduler can be layered before
  it; co-optimizing schedule + key liveness is future work, and the DFA
  gating keeps the transformation correct for whatever order it is given.

### 2.3 Prefetch insertion policy (answers C4 for reviewer A, rebuttal Q4)

Pass: `lib/Dialect/KMRT/Transforms/KeyPrefetching.cpp`; options in
`KeyPrefetching.td:118-127`.

- **Cost-model-based distance, not a fixed op count.** Option
  `prefetch-threshold` (default **50** cost units). Per-op costs are
  hard-coded (`getOperationCost`, `:219-239`): Bootstrap=100, Chebyshev=50,
  Rot=15, KeySwitch=12, Mul=10, Relin=8, Add=1, key-management ops=0. The
  pass walks backwards from each `load_key` accumulating cost until the
  threshold is reached, and inserts `prefetch_key` there (`:679-790`),
  rematerializing the key index at the insertion point.
- **Loops: software-pipelined prefetch.** First iteration's key is
  prefetched before the loop; inside the body, iteration `iv+step`'s key is
  prefetched under an `affine.if (iv+step < ub)` guard (`:1400-1578`).
  A cost-based variant places a single prefetch at iteration
  `iterCount - ceil(threshold/iterCost)` (`:991-1075`). Post-loop key groups
  are batched and spread across preceding loop iterations to smooth load on
  the deserializing thread (`:320-570`).
- **Runtime-delegated mode** (`runtime-delegated=1`, used for Balanced mode
  per `README.md:87-93`): all prefetches are emitted as an ordered prologue
  mirroring execution order (`:2071-2375`), and the runtime paces them.
- **Verification mode**: an abstract-interpretation check that every
  `load_key` is covered by a `prefetch_key` (`:2026-2052`) — usable as
  evidence of policy correctness in the rebuttal.
- BSGS loops need no special casing — they are consumed as generic nested
  affine loops (test: `tests/Dialect/KMRT/Transforms/key_prefetching_bsgs.mlir`).

### 2.4 Runtime side (C2, C3, C4-runtime, C6) — facts from the runtime repo

The runtime lives in the companion repo `eymay/KeyMemRT`
(`include/KeyMemRT.hpp`, single header, ~1550 lines). The emitter calls its
API: `enqueueKey(index[,depth])` for prefetch
(`OpenFhePkeEmitter.cpp:1398-1421`), `deserializeKey(index)` for load
(`:1299`), `clearKey` (`:1394`), `serializeAllKeys` / `serializeKeysAtLevel`
/ `clearAllKeys`. Runtime internals (all citations into
`KeyMemRT/include/KeyMemRT.hpp`):

- **Four runtime modes** (`KeyMemMode`, `:54-59`): `IGNORE` (all key ops are
  no-ops; keys stay resident — the ANT-ACE-like baseline configuration),
  `IMPERATIVE` (synchronous on-demand deserialize — Low-Memory mode),
  `PREFETCH` (background deserialization thread + queue — Balanced mode),
  and `SPECULATIVE` (blocks polling for the key *file* to arrive — models
  cold-start/network-transfer scenarios; useful against reviewer B's
  "network costs untouched" point as ongoing work).
- **One background deserialization thread**, started only in PREFETCH mode
  (`startPrefetchMode`, `:368-376`; worker loop `deserializationWorker`,
  `:1358-1483`). FIFO `deserializationQueue` of (rotationIndex, depth) pairs
  guarded by mutex + condition variable; `enqueueKey` pushes and notifies
  (`:1327-1356`).
- **Explicit memory budget**: the prefetcher is capped by a *tower budget*
  (`prefetchTowerLimit`, CLI flag `--prefetch-sat`, default 250 towers,
  `:95,:133-136,:1540`). The worker only pops the next queue item if
  `currentTowerCount + requiredTowers <= prefetchTowerLimit` (`:1383,
  :1417`); otherwise it sleeps until `clearKey`/`clearAllKeys` frees budget
  and notifies it (`:960-967, :1198-1212`). So Balanced mode **does bound
  prefetched-key memory** — the budget is in RNS-tower units, i.e.
  proportional to bytes, not a naive key count.
- **Blocking semantics**: if `deserializeKey` finds the key not yet ready in
  PREFETCH mode, it self-enqueues if missing from the queue and waits on a
  condition variable, logging the wait time in microseconds (`:563-698`,
  wait-time log `:597-599, :629-631`). A watchdog detects
  budget-too-small deadlock (2-minute timeout) and aborts with an actionable
  "increase --prefetch-sat" message (`:651-689`).
- **Serialization format**: per-key binary files
  (`rotation_key_<idx>[_l<depth>].bin`, `:877-886`) written/read with
  OpenFHE's `SerializeEvalAutomorphismKey` / `DeserializeEvalAutomorphismKey`
  (`SerType::BINARY`, `:527-528, :838-839`). `clearKey` erases exactly one
  automorphism key from OpenFHE's key map (`:906-923`).
- **Key compression** (`KeyCompression.hpp`): before serialization, keys can
  be compressed to their profile-derived level — dropping Q towers to
  `multDepth + 2 - level` (`KeyMemRT.hpp:1006-1016`,
  `KeyCompression.hpp:67-99`) — shrinking both on-disk size and
  deserialization work; restored via `RestoreDynamicQSize` after load
  (`:848-861`). This is the "kc" benchmark configuration in the Justfile.
- **Per-network budget values** used in evaluation (`KeyMemRT/Justfile:227-239`):
  `--prefetch-sat` 600 (MLP, LoLa), 2000 (ResNet-1, LeNet), 2500
  (ResNet-8/10/20/32/44/56), 3500 (AlexNet, ResNet-18/34), 4500
  (ResNet-50, VGG, YOLO). These answer "how is the user-provided limit
  selected" concretely, and sweeping this knob is exactly the
  memory–latency curve reviewer C asks for.
- Honest caveat: `calculateTowerCount` is currently a stub returning a flat
  32 towers per key (`:414`) — the budget bounds (#prefetched keys × 32)
  tower units rather than the exact per-depth tower count
  (`getActualTowerCount`, `:1514-1518`, exists but is not used in the
  accounting). Worth fixing before the revision.

Mode mapping across the two repos: Low-Memory = compiler pipeline without
prefetching + runtime `--key-mode imperative`; Balanced = pipeline with
`--kmrt-key-prefetching="runtime-delegated=1"` + runtime
`--key-mode prefetching --prefetch-sat N`; the ANT-ACE-style all-resident
baseline = `--key-mode ignore` with single-file bulk key deserialization
(`deserializeAllKeysFromSingleFile`, `:1093-1128`; Justfile `run-*-server`
targets).

**Measurement infrastructure that already exists** (runtime repo):
- PREFETCH wait times per key are logged in µs (`KeyMemRT.hpp:597-599`) —
  directly usable to report how much latency prefetching hides (C-Q3).
- `ResourceMonitor.hpp` samples every 10 ms: process RSS/VMS/heap/peak/PSS
  (`/proc/self/status`, `smaps_rollup`; `:271-314`) and, with advanced stats,
  **system cached/buffers GB, page-ins/outs, and major faults**
  (`:359-425`) — i.e. the page-cache accounting reviewer E says is missing
  is already instrumented; the rebuttal can report cached-memory curves.
- Named event markers with per-thread start/end timestamps
  (`mark_event_start/end`, `:79-106`) exported to CSV — the hook needed for
  the deserialize-vs-I/O split (A-Q3); currently used for serialization
  events in `drivers/client_driver.cpp:259-322`, needs to be added around
  the `DeserializeEvalAutomorphismKey` call to separate file-read from
  deserialization time.

### 2.5 Bootstrap removal and its correctness (C7)

- `UnnecessaryBootstrapRemoval`
  (`lib/Transforms/UnnecessaryBootstrapRemoval/UnnecessaryBootstrapRemoval.cpp`)
  removes a `bootstrap` only when the profiled tower count of its input
  equals that of its output (`:70`), i.e. the bootstrap performs no level
  reset. Bootstraps without complete annotations are skipped (`:69,:82-86`).
- The `result_towers` annotations come from `ProfileAnnotator`
  (`lib/Transforms/ProfileAnnotator/ProfileAnnotator.cpp`), which ingests a
  runtime profile (`PROFILE::OUTPUT:...:TOWERS:...`) and stamps it on the IR.
  So the pass is **profile-guided**, not backed by static level/noise
  analysis. Rebuttal framing: a bootstrap whose input already has the towers
  the bootstrap would produce is a no-op by CKKS semantics (it cannot raise
  the level further), so removal preserves the level schedule observed at
  runtime; end-to-end accuracy validation on the benchmark outputs is the
  right evidence to add (reviewer D's request) and cheap to produce.
- Why it is in this work at all: `bootstrap-rotation-analysis`
  (`lib/Transforms/BootstrapRotationAnalysis/BootstrapRotationAnalysis.cpp`)
  expands each bootstrap into its constituent internal rotation keys
  (reproducing OpenFHE's CoeffsToSlots/SlotsToCoeffs BSGS index math,
  `:197-429`) and emits per-index `load_key`/`clear_key` with per-key depth
  from a config registry (`:23-180`, `:635-708`). Bootstrap keys dominate
  the key footprint, so exposing them — and eliminating bootstraps that
  contribute keys without contributing levels — is integral to key-memory
  management, not an orthogonal bolt-on. This is the argued connection;
  the *quantitative* separation still needs the ablation (C1).

### 2.6 Other pipeline passes and their separability (C1)

- **BSGS decomposition** (`lib/Transforms/SymbolicBSGSDecomposition/…`):
  rewrites an IV-dependent rotation loop into prologue/main/epilogue loops;
  baby-step keys (N2 = ceil(sqrt(range))) are loaded once into a memref and
  reused via `use_key`; one giant-step key is live at a time and cleared
  immediately. Working set drops from R keys to ~sqrt(R)+1. Note this pass
  *is* key-lifetime management (it emits the KMRT ops directly) — in the
  ablation narrative, BSGS decomposition and lifetime management are not
  fully orthogonal, and that is worth saying explicitly.
- **Rotation hoisting** is HEIR/OpenFHE's `openfhe-fast-rotation-precompute`
  (`lib/Dialect/Openfhe/Transforms/Passes.td:123-134`) — a pre-existing
  technique the pipeline reuses (one shared digit decomposition, many fast
  rotations). Cite it as reused, not contributed.
- **Ciphertext/plaintext clearing**
  (`lib/Dialect/Openfhe/Transforms/InsertClearOps.cpp`) uses MLIR `Liveness`
  to insert `clear_ct`/`clear_pt` after last use, with conservative
  exclusions (function args, loop-defined and returned values; clears for
  in-loop last uses are hoisted after the outermost loop). It touches only
  ciphertext/plaintext data and has zero coupling to KMRT key ops — cleanly
  ablatable.

### 2.7 Facts for the fairness question (C2)

The evaluation harness (`KeyMemRT/Justfile`) runs **every** configuration —
baseline (`ignore`), Low-Memory (`imperative`), key-compression, Fhelipe,
and Balanced (`prefetching`) — with `OMP_NUM_THREADS=4`
(`Justfile:123-147, :200, :208, :216, :241, :251`). So the OpenFHE baseline
already gets 4 OpenMP compute threads; reviewer A's premise that the
baseline "appears to run without the same threading advantage" is factually
answerable: compute-thread counts are equal across all configurations.
Balanced mode adds exactly **one** extra thread (the deserialization worker,
`KeyMemRT.hpp:374`), which performs no FHE computation — only the disk I/O +
deserialization that the baseline pays on the critical path at startup.
The clean remaining check (run the baseline with `OMP_NUM_THREADS=5`) is a
cheap experiment worth adding, since OpenFHE's OpenMP scaling on these
kernels is sub-linear.

---

### 2.8 Orion (E-Q1)

The runtime repo's own flow **uses Orion as a frontend**: benchmarks are
built via the Orion FHE compiler plus `orion_heir_translator`
(`KeyMemRT/README.md`, Dependencies step 3;
`benchmarks/networks/*.py` are Orion network definitions). This
strengthens the comparison story: KeyMemRT consumes Orion-generated
programs, so a head-to-head against Orion's hand-written coarse-grained key
loading is natural — same frontend, hand-crafted vs. compiler-automated
per-key management — and the qualitative distinction (Orion loads
coarse-grained groups, hand-placed per model; KeyMemRT derives per-key
lifetimes automatically and prefetches under an explicit budget) can be
stated from direct experience with the Orion flow.

## Part 3 — What must come from experiments (cannot be cited from either repo)

1. **Ablation matrix (C1)** — per-pass on/off runs: merge pass, prefetching,
   BSGS, bootstrap removal, clear-ops; plus the Low-Memory-beats-ANT-ACE
   explanation (likely candidates: allocator/page pressure from 50+ GB
   resident keys, avoided key-generation-time setup, clear_ct reducing
   ciphertext footprint — needs measurement, not speculation).
2. **Key movement microbenchmark (C3, A-Q3)** — per-key timing split: disk
   read vs deserialization vs context insertion. Most instrumentation
   already exists (PREFETCH wait-time µs logs, `ResourceMonitor` event
   markers and cached/major-fault sampling — see 2.4); what's missing is an
   event marker around `DeserializeEvalAutomorphismKey` separating file read
   from decode. Run with page cache dropped
   (`echo 3 > /proc/sys/vm/drop_caches`) and with `cgroup` memory caps to
   kill the 512 GB page-cache objection (E). Report storage medium and
   bandwidth. Also report the key-compression ("kc") effect on per-key load
   time — compressed keys shrink both file size and decode work.
3. **Constrained-memory demonstration (D, C6, C-Q5)** — run ResNet variants
   under cgroup limits where the all-resident baseline OOMs and KeyMemRT
   completes; sweep `--prefetch-sat` (the knob already exists, per-network
   values 600–4500 towers) to trace the memory–latency curve. This directly
   answers "characterize the trade-off under different memory budgets".
4. **Threading fairness run (C2/A-Q2)** — all configs already run with
   `OMP_NUM_THREADS=4` (see 2.7); add a baseline run at
   `OMP_NUM_THREADS=5` to match KeyMemRT's total thread count, and a
   Balanced run with the worker disabled (imperative mode) to isolate
   prefetch-overlap benefit from key-liveness benefit.
5. **Compile-time overhead (E-Q2)** — wall-clock of the KMRT passes on
   ResNet-50 IR; also report inserted load/clear/prefetch op counts per
   benchmark (cheap: `keymemrt-opt` + `grep -c`).
6. **Orion comparison (E-Q1)** — Orion is already the frontend (see 2.8);
   run at least one shared benchmark against Orion's own hand-crafted key
   loading for a quantitative point.
7. **Numerical accuracy (D)** — output precision (log2 error) with and
   without bootstrap removal on each benchmark.

## Part 4 — Suggested rebuttal skeleton (per reviewer question)

- **A-Q1 (Low-Memory faster than ANT-ACE)**: acknowledge; give measured
  explanation from ablation item 1; note ANT-ACE holds ~50 GB of keys
  resident, which taxes the allocator/TLB even in all-RAM mode.
- **A-Q2 (thread fairness)**: cite 2.7 facts (all configs run with
  `OMP_NUM_THREADS=4`; the worker thread does no FHE compute) + run item 4.
- **A-Q3 (deserialization cost)**: run item 2; report per-key ms split;
  cite existing µs wait-time logging (2.4).
- **A-Q4 (prefetch distance)**: cite 2.3 — cost-model threshold (default 50
  weighted ops), software-pipelined next-iteration prefetch in loops,
  verification pass guaranteeing coverage.
- **A detailed feedback (runtime prefetch policy)**: cite 2.4 — FIFO queue +
  single worker, tower-budget admission control (`--prefetch-sat`),
  blocking load with self-enqueue fallback, deadlock watchdog.
- **C-Q1/D-Q1 (merge policy, loops)**: cite 2.1/2.2 — DFA + per-category
  windows (10 FHE-ops sequential, 20/50/100 for pre/post-loop) + affine-set
  analysis for loops; commit to making thresholds pass options in revision.
- **C-Q2 (schedule sensitivity)**: cite 2.2 — post-optimizer on a fixed
  order, conservative DFA; add worst-case discussion; future work.
- **C-Q3 (movement overhead / caching)**: run item 2.
- **C-Q4/D (ablation, bootstrap correctness)**: run items 1 and 8; cite 2.5
  for the mechanism and the integral-to-key-memory argument.
- **C-Q5 (budget sweep)**: cite 2.4 — `--prefetch-sat` is an explicit
  tower-unit memory budget with per-network values already tuned
  (600–4500); run item 3 for the curve.
- **C (why runtime info at all)**: the compiler knows key-use order but not
  the machine's I/O speed or concurrent memory pressure; runtime-delegated
  mode keeps the compiler-derived order while the tower-budget admission
  control adapts pacing to the deployment machine — a static schedule would
  bake in one machine's latencies. Cite 2.3 (runtime-delegated) + 2.4.
- **E-Q1 (Orion)**: cite 2.8 (Orion is the frontend) + run item 6.
  **E-Q2 (compile time)**: item 5.
- **E (page-cache / memory accounting)**: cite 2.4 — `ResourceMonitor`
  already samples system cached/buffers, page-in/out, and major faults
  alongside process RSS/PSS; report those curves + run item 2's
  drop-caches/cgroup runs.
- **B (framing)**: no code needed — soften "unlock" claim, add Fhelipe
  points to Fig. 1, cap the trend line at the 2·sqrt(N) key-count ceiling,
  and scope claims to RAM-residency (concede generation/transfer/disk costs
  are unaffected).
