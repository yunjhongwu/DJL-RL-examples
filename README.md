DJL-RL
======
Examples of reinforcement learning implementations with [DJL](https://djl.ai/) (only tested with [PyTorch](https://pytorch.org/) 1.6 backend). This repository aims to provide toy examples of RL models in Java. All the implementations are always tested with nightly builds of DJL, which is still under active development. They may break occasionally/not be the best practice.

For debug and benchmark purpose, the repository also includes a DynaQ agent, which does not rely on DJL, two [Gym](https://gym.openai.com/) tasks, and a http client for subscribing external environment.

 - Models:
   - DQN
   - Advantage Actor Critic (A2C)
   - Quantile Regression DQN (QRDQN)
   - Generalized Advantage Estimation (GAE)
   - Proximal Policy Optimization (PPO)
   

 - Gym environments:
   - CartPole
   - MountainCar
 

