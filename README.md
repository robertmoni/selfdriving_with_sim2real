# Self Driving Agent training with Simulation-to-Real trasnfer using Python, Ray RLlib and gym-duckietown

This repository contains the steps of a tutorial to succesfully train a Reinforcement Learning agent using Domain \\
Randomization or Daomain Adaptation and Deep Reainforcement Learning to train an agent for self-driving in the Duckietown Gym environment.

![]("art/tools.png")

# Outline 
[1. Setup Conda Environment](#setup_conda_environment)


[2. Manaul Control example](#manual_control)


[3. Tutorial 1: Training with Domain Randomization](#tutorial_dr)


[4. Tutorial 2: Training with Domain Adaptation](#tutorial_da)


[5. Testing in the Duckietown Gym Simulator](#testing_agent)

![]("art/concept.png")
## 1. Setup Conda Environment
Run conda environment setup:
```$ bash setup_conda_environment.sh```

Run jupyter notebook:


## 2. Manual Control example
#manual_control

## 3. Tutorial 1: Training with Domain Randomization
![]("art/just_policy.png")
#tutorial_dr


## 4. Tutorial 2: Training with Domain Adaptation

#tutorial_da


## 5. Testing in the Duckietown Gym Simulator
#testing_agent



```nvidia-docker run --rm -v /home/robertmoni/projects/selfdriving_with_sim2real:/home/selfdriving_with_sim2real -td -p 2249:22 -p 7080:6006 -p 7081:8888 -w /home/general/  --name dockerrm rmc26/selfdriving_with_sim2real```

## 2. Jupyter

```jupyter notebook --no-browser  --port 8805 --ip 0.0.0.0```
