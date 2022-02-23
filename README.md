# SR-GNN

This is a pytorch implementation of AAAI'19 paper [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)

Running on single GPU:
```bash
python main.py
```
### Performance
- Result was slightly better than original repo on sample dataset. 
- This repo also uses the exact network architecture proposed in the paper, which has a different GNN aggregation logic to the official repo.
- Speed is slightly slower than the original repo because of log file writing.
