# TradingDesk — Born Traders

Project 2: Custom local language models that replace API agents one by one.

## Architecture

```
factory/          # Shared, reusable pipeline (tokenizer, model, training, inference)
agents/           # Per-agent configs, training data, and weights
docs/decisions/   # Architecture and infrastructure decisions
```

## First Agent: Data Manager

Training pipeline for the Data Manager Born Trader.
Run: `python factory/training/train.py --agent data_manager`

## TDDEV-9 — Board Decision 19 April 2026
