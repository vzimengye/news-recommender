# News Recommender

A learning project for building a reproducible two-stage news recommendation system with Microsoft's MIND-small dataset. The planned system has a retrieval stage, a ranking stage, and a small FastAPI service. Work is being developed incrementally; only completed and measured components will be reported as results.

## Project status

The repository is at the data and project setup stage. The implementation and evaluation tables will be updated as each milestone is completed; see "Planned system" below for the roadmap.

## Dataset

This project uses MIND-small, an English news recommendation research dataset released by Microsoft Research. Before using it, read and comply with the [Microsoft Research License Terms](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf). The dataset is not included in Git because it is distributed under its own terms.

Expected local layout:

```text
data/raw/
├── MINDsmall_train.zip
├── MINDsmall_dev.zip
├── MINDsmall_train/
│   ├── behaviors.tsv
│   └── news.tsv
└── MINDsmall_dev/
    ├── behaviors.tsv
    └── news.tsv
```

Download the two archives from the [official MIND download page](https://msnews.github.io/) after accepting the terms. Extract each archive into its matching directory. Keep the original archives locally for reproducibility; `data/` is ignored by Git.

The official `dev` split is a held-out evaluation set in this project. Any tuning split made from `train` must respect time ordering. The final protocol will be documented alongside the reported metrics.

## Planned system

1. Parse impression logs into candidate-level examples.
2. Establish a popularity retrieval baseline and a simple logistic-regression ranker.
3. Compare item-based collaborative filtering and text-embedding retrieval.
4. Evaluate ranking models and add serving, experiment tracking, and monitoring.

Planned methods are not necessarily implemented yet. Metrics will be reported only after they have been run and checked.

## Repository layout

```text
configs/       Configuration files
data/          Local source and derived data (ignored by Git)
docs/          Project plan and design notes
notebooks/     Exploratory analysis
scripts/       Command-line entry points
src/           Reusable application and ML code
tests/         Automated checks
```

## Development

Python 3.11 is the planned runtime. Environment setup and executable commands will be added as the first code milestone is implemented.

## Citation

If you use MIND in research, cite:

> Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, and Ming Zhou. “MIND: A Large-scale Dataset for News Recommendation.” ACL 2020.
