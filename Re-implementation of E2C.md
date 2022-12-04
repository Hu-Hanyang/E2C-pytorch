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



```python
python train_e2c.py --env=planar --propor=0.1 --batch_size=32 --lr=0.001 --num_iter=1000 --iter_save=800 --log_dir='hanyang_try' --seed=2
```

<img src="/localhome/hha160/projects/E2C-pytorch/debug1.png" style="zoom:50%;" />



python train_e2c.py --env=planar --propor=0.01 --batch_size=32 --lr=0.001 --num_iter=20 --iter_save=16 --log_dir='hanyang_try' --seed=2

2022/12/3
python train_e2c.py --env=planar --propor=0.75 --batch_size=128 --lr=0.0001 --num_iter=5000 --iter_save=1000 --log_dir='noraml_train1' --seed=3047
