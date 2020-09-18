DJL-RL
======
Examples of reinforcement learning implementations with [DJL](https://djl.ai/) (only tested with [PyTorch](https://pytorch.org/) 1.6 backend). This repository aims to provide toy examples of RL models in Java. 

This repo is always tested with nightly builds of DJL, which is still under active development. The implementations in this repo may break occasionally/not be the best practice.

 - Models:
   - DQN
   - Advantage Actor Critic (A2C)
   - Quantile Regression DQN

DynaQ, which does not rely on DJL, is also provided for debug and benchmark purpose.
   
The repository also includes two easy [Gym](https://gym.openai.com/) tasks to help users debug. 

 - Gym environments:
   - CartPole
   - MountainCar

