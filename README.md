# Procedural Game Balancing Thesis

Compact project page for my Honours Computer Science thesis on procedural game balancing.

The project explores how generated game content can be evaluated automatically using a deterministic simulation backend. I built an experiment pipeline around a frozen open-source Super Auto Pets simulator to test generated artifacts for fairness, diversity, stability, and safety before comparing them across reproducible runs.

## Highlights

- Built a deterministic Super Auto Pets simulation platform with reproducible seeding, action validation, and generated-content injection.
- Created config-driven experiment workflows for baseline, generator, iterative, and transfer runs.
- Added checkpointing and cached evaluations so large experiment batches could be resumed and compared consistently.
- Implemented simulator-backed metrics for fairness, diversity, stability, and safety.
- Automated post-run analysis into thesis-ready tables and figures.

## Tech Stack

Python, NumPy, PyYAML, PyTest, Git

## Repository Layout

```text
.
├── README.md
├── docs/
│   ├── project-summary.md
│   └── results-notes.md
├── configs/
│   └── example-experiment.yaml
├── figures/
│   └── .gitkeep
└── src/
    └── .gitkeep
```

## Status

This repository is a short public-facing summary for resume review. Source code, experiment configs, thesis PDF, and selected figures can be added here as they are cleaned for public release.

## Resume Description

Built a deterministic Super Auto Pets simulation platform with reproducible seeding, generated-content injection, config-driven experiments, checkpointed evaluations, and simulator-backed fairness, diversity, stability, and safety metrics.
