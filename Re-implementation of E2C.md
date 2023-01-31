## Re-implementation of E2C

### Hanyang Hu
### Visualize the training results: tensorboard --logdir=logs

## 1. planar
### data generation
python sample_planar_data.py --sample_size=3000
### training
1. python train_e2c.py --env=planar --propor=0.75 --batch_size=128 --lr=0.0001 --num_iter=5000 --iter_save=1000 --log_dir='normal_train1' --seed=3047

2. python train_e2c.py --env=planar --propor=0.75 --batch_size=128 --lr=0.0001 --num_iter=5000 --iter_save=1000 --log_dir='normal_train2' --seed=3047
3. python train_e2c.py --env=planar --propor=0.75 --batch_size=128 --lr=0.0001 --num_iter=5000 --iter_save=1000 --log_dir='normal_train3' --seed=3047 # adding new method in E2C model

## 2. cartepole
### data generation
python sample_cartpole_data.py --sample_size=15000
### training
python train_e2c.py --env=cartpole --propor=0.75 --batch_size=128 --lr=0.0001 --lam=1 --num_iter=5000 --iter_save=1000 --log_dir='cartpole_train' --seed=3047

## 3. pendulum
### data generation
python sample_pendulum_data.py --sample_size=15000 # failed
AttributeError: 'PendulumEnv' object has no attribute 'step_from_state'
### training
python train_e2c.py --env=pendulum --propor=0.75 --batch_size=128 --lr=0.0003 --lam=1 --num_iter=5000 --iter_save=1000 --log_dir='normal_train2' --seed=1997
