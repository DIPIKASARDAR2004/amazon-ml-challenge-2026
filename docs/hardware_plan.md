# Hardware and execution plan

Updated 2026-09-26 after Pass 4. **Start Pass 5 locally with a small training sample. Moving to Kaggle is optional; no measured result currently requires it.** Real classifier training memory has not yet been measured. This is resource planning, not a completed Pass 5 run.

**There is no measured 30–50 GiB storage requirement or 56-hour next pass.** The current dataset directory uses about 2.7 GiB and retrieval artifacts about 4.1 GiB. The 55.5-hour figure below is a linear extrapolation for sequential retrieval over all 1.73 million test businesses, not the next sampled experiment or a measured full run. At the same rate, retrieval for 5,000 training businesses would take about ten minutes; feature construction and training time remain unmeasured.

## What this environment actually exposes

| Resource | Observed value | Interpretation |
|---|---:|---|
| Physical laptop RAM | **16 GB**, confirmed by the user's Windows screenshot | Windows and WSL share this physical memory |
| WSL2 RAM | About 7.6 GiB total; 6.0 GiB available at inspection | Available memory changes as applications run |
| Swap | 2.0 GiB | Disk-backed overflow; heavy swapping can make work very slow |
| CPU | 8 logical CPUs, Intel Core Ultra 5 226V | Current retrieval processes queries sequentially; the fixture trainer explicitly uses one thread |
| Linux filesystem | About 923 GiB reported available | This is the WSL virtual filesystem's view, not verified Windows host free space |
| Physical storage shown by Windows | 477 GB total, 287 GB used; approximately **190 GB free** | Use the physical host's remaining space for disk planning |
| Local dataset directory | About 2.7 GiB | Includes original and derived data files |
| Retrieval artifacts directory | About 4.1 GiB | Mostly the complete training target index |

The user's screenshot confirms **16 GB of physical RAM**. The 7.6 GiB reading is the memory visible inside WSL. Microsoft documents a default WSL2 limit of 50% of Windows RAM, so the observed Linux capacity is consistent with the default; the actual host `.wslconfig` has not been inspected. The WSL limit can be adjusted if a measured workload needs more room, but Windows and the IDE also need memory. The first small Pass 5 run does not currently justify changing it. No WSL configuration was changed. [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#configuration-settings-for-wslconfig).

WSL uses a virtual disk. Its reported capacity can exceed the currently available space on the physical Windows drive. Here the screenshot shows approximately **190 GB of physical storage free**, not the 923 GiB shown by the Linux filesystem. Recheck Windows free space as artifacts grow. [Microsoft WSL disk-space guidance](https://learn.microsoft.com/en-us/windows/wsl/disk-space).

## RAM, disk, CPU, and GPU are separate limits

**RAM is the active workspace.** Keeping millions of Python objects, duplicate text columns, candidate pairs, and model-training buffers at once can exhaust it. A 4 GiB database on disk does not require 4 GiB of process RAM: our index build peaked at 232.13 MiB, and the two 1,000-query retrieval runs peaked at 225.85 / 264.98 MiB. These are process-RSS measurements, not total machine memory including the OS filesystem cache.

**Disk stores data and artifacts.** Raw files, the target database, feature shards, outputs, and checkpoints consume storage. Index construction and sorting also need temporary space. The final test target pool will need its own index; the training index cannot be reused as the test target pool.

**CPU time controls how long text processing and search take.** SQLite lookup, Python pair construction, and the current feature path run on the CPU. Eight visible CPUs do not imply an automatic eightfold speedup; parallel execution must be designed, measured, and balanced against disk contention and duplicated memory.

**GPU memory is separate from ordinary system RAM.** Selecting a GPU notebook does not give SQLite or ordinary Pandas operations access to GPU acceleration automatically. Our current model path uses LightGBM's default CPU device. GPU training requires a supported build and configuration, and it would not remove the CPU retrieval work. [LightGBM device/thread documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#device_type).

## Estimate pair counts before choosing hardware

The compact development run averaged 92.185 candidates per Source 1 business. Applying that density elsewhere is a planning assumption; actual training/test density may differ. The current feature function produces ten float64 numbers per pair, or 80 bytes of numeric payload.

| Query businesses | Projected pairs | Numeric feature payload only | Projected retrieval time only |
|---|---:|---:|---:|
| 5,000 training sample | 460,925 | 35.2 MiB | 9.6 minutes |
| 50,000 | 4.61 million | 351.7 MiB | 1.6 hours |
| 100,000 | 9.22 million | 703.3 MiB | 3.2 hours |
| All 1,986,139 training-partition businesses | 183.09 million | 13.64 GiB | 63.7 hours |
| All 1,732,544 competition test businesses | 159.71 million | 11.90 GiB | 55.5 hours |

**The feature column is not a total-RAM estimate.** It excludes IDs, strings, Python lists/tuples/dictionaries, DataFrame indexes, temporary copies, labels, and model-training buffers. The current small-data feature function also accumulates Python rows before creating its numerical table. Loading every training pair through that path would exceed this machine's RAM. Even a 30 GiB cloud runtime is not a guarantee for that approach.

Times extrapolate the measured compact search rate of 0.1154 seconds per business on this machine. They exclude feature calculation, model fitting/scoring, index construction, and output writing. Caches, query composition, hardware, and concurrency can change them substantially. They show the scale of the work, not predicted completion times.

Inference can score one batch and discard its feature matrix before loading the next. Therefore 160 million total test pairs do not need to reside in RAM together. Supervised training needs a separate sample/materialization strategy; reading input in chunks does not by itself make a model trained on every pair memory-free. We do not need to begin by training on all 183 million projected pairs.

## Kaggle and Colab tradeoffs

Kaggle's documentation currently lists CPU notebooks with **4 CPU cores, 30 GB RAM, 12-hour execution sessions, and 20 GB of saved working-directory storage**. Additional scratch storage is session-only. Verify the allocation shown in the actual session. It offers useful RAM headroom, but these specifications do not establish that its CPU or storage will be faster than this laptop. [Kaggle notebook specifications](https://www.kaggle.com/docs/notebooks#technical-specifications).

Colab's official FAQ says resource availability/limits vary; free notebooks can run for at most 12 hours depending on availability and usage. High-memory paid runtimes are also subject to availability. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

For this project, consider a Kaggle **CPU** notebook when measured model-training RAM becomes the limiting factor. An accelerator session is not necessary for the current algorithm. First verify Python/package compatibility and SQLite FTS5 support: the repository's pinned environment was tested locally with Python 3.14.4, not on Kaggle or Colab.

Both notebook platforms require deliberate persistence of outputs and resumable batches for work exceeding a session's lifetime. The current rough full-test search estimate is much longer than one 12-hour notebook session. Migrating alone does not solve that runtime constraint.

Moving data and a roughly 4 GiB index also costs transfer/setup time. Keep an active SQLite database on the execution machine's local storage; copy completed artifacts to persistent storage rather than performing every random lookup through a remote drive mount. Google's FAQ recommends reducing mounted-Drive I/O for performance. [Colab storage guidance](https://research.google.com/colaboratory/faq.html).

## Recommended next steps

1. Implement a measured Pass 5 pilot locally, starting with a deterministic subset of training queries and then the existing 5,000-query training sample. Continue searching the full target pool.
2. Batch feature construction and avoid retaining repeated names/addresses in every pair record. Keep detailed text traces for inspection samples rather than every future pair.
3. Measure end-to-end time and peak process memory for retrieval, features, and model fitting separately. Use an initial planning budget of roughly 3–4 GiB for the job, leaving headroom in the currently available WSL memory. This is a budget, not a proven requirement.
4. Expand training only when development results justify the extra pairs. Check class balance and label correctness rather than treating all available negative pairs as automatically useful.
5. Before full inference, implement persistent batches/checkpoints and benchmark modest CPU parallelism. More workers consume more memory and may compete for disk I/O.
6. Move to a larger CPU/RAM machine if the measured workload warrants it. A 16–32 GiB CPU environment is a reasonable capacity tier for larger sampled experiments, not a requirement or purchase recommendation. Long production runs also need persistent storage and predictable runtime, not merely more RAM.

An earlier **30–50 GiB** storage allowance was speculative headroom for possible separate train/test indexes, temporary work, feature files, and outputs. It is not a measured requirement or a minimum needed to begin Pass 5. Measure actual growth as those stages are implemented; avoid retaining verbose raw-text traces for every future pair.

Current decision: **continue locally for the first Pass 5 baseline; profile before migrating or buying hardware.** Training and full-scale inference remain unimplemented, so their final resource requirements are still open measurements.
