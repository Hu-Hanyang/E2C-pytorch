## Re-implementation of E2C

### Hanyang

### 1. Data generating
python sample_{env_name}_data.py --sample_size={sample_size}
1. planar: python sample_planar_data.py --sample_size=3000; finished;
2. cartpole: python sample_cartpole_data.py --sample_size=15000; finised.
3. pendulum: python sample_pendulum_data.py --sample_size=15000 # AttributeError: 'PendulumEnv' object has no attribute 'step_from_state'

### 2. Training:

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

2022/12/4
python train_e2c.py --env=planar --propor=0.75 --batch_size=128 --lr=0.0001 --num_iter=5000 --iter_save=1000 --log_dir='noraml_train2' --seed=1997

2022/12/5
pendulum data: N = 15,000
python sample_pendulum_data.py --sample_size=15000
python train_e2c.py --env=pendulum --propor=0.75 --batch_size=128 --lr=0.0003 --lam=1 --num_iter=5000 --iter_save=1000 --log_dir='noraml_train2' --seed=1997

cartepole 
python sample_cartpole_data.py --sample_size=15000
python train_e2c.py --env=cartpole --propor=0.75 --batch_size=128 --lr=0.0001 --lam=1 --num_iter=5000 --iter_save=1000 --log_dir='cartpole_train' --seed=3047
