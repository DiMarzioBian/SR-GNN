# SR-GNN


Original paper: https://arxiv.org/abs/1811.00855

### This is a pytorch replicate of SR-GNN

Running on single GPU:
```bash
python main.py
```
### Performance
- Result was slightly better than original repo on sample dataset. 
- This repo also uses the exact network architecture proposed in the paper, which has a different GNN aggregation logic to the official repo.
- Speed is slightly slower than the original repo because of log file writing.