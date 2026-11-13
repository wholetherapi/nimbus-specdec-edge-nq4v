# Latency-Aware Speculative Decoding for Edge Translation

Official implementation of **Latency-Aware Speculative Decoding for Edge Translation** (ACL 2026).

The repository contains the latency-aware speculative decoder described in the
paper: a draft-depth scheduler driven by an offline latency model, plus training
and evaluation entry points used for the reported experiments.

## Layout

```
src/specdec/    decoder, draft model and latency scheduler
scripts/        training and evaluation entry points
configs/        experiment configuration
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/train_draft.py --config configs/edge_en_de.yaml
python scripts/evaluate.py --config configs/edge_en_de.yaml --split test
```

## Citation

If you use this code, please cite the ACL 2026 paper.
