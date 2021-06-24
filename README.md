# Self Driving Agent training with Simulation-to-Real trasnfer using Python, Ray RLlib and gym-duckietown

This repository contains the steps of a tutorial to succesfully train a Reinforcement Learning agent using Domain \\
Randomization and Deep Reainforcement Learning to train an agent for self-driving in the Duckietown Gym environment.

![](art/tools.png")

# Outline 
[1. Setup Conda Environment](#setup_conda_environment)


## 1. Setup Conda Environment
Run conda environment setup:
```$ bash setup_conda_environment.sh```

Run jupyter notebook:


## 2. Manual Control example
#manual_control

## 3. Tutorial -Training setup & Training
#tutorial


## 4. Testing
#testing_agent

## 1. Docker init  

```nvidia-docker run --rm -v /home/robertmoni/projects/selfdriving_with_sim2real:/home/selfdriving_with_sim2real -td -p 2249:22 -p 7080:6006 -p 7081:8888 -w /home/general/  --name dockerrm rmc26/selfdriving_with_sim2real```

## 2. Jupyter

```jupyter notebook --no-browser  --port 8805 --ip 0.0.0.0```
