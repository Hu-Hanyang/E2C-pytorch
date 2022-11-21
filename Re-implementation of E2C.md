# Re-implementation of E2C

### Hanyang

#### Terminal

1. Data generating

2. Training:

```python
python train_e2c.py --env=planar  --batch_size=32 --lr=0.0001 --lam=0.25 --num_iter=5000 --iter_save=1000 --log_dir=trying --seed=2
```

bug:

RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED