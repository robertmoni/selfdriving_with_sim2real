# Self Driving Agent training with Simulation-to-Real transfer learning methods

![Tools](art/tools.png)

This repository contains a brief tutorial on training a Reinforcement Learning agent for Lane Following in the Duckietown environment.
The agent is trained using the Proximal Policy Optimization (PPO) algorithm with Generalized Advantage Estimation (GAE) method in an Actor-Critic framework.
Since the Duckietown environment supports deployment into real life environment with Duckiebot robots, we provide our methods with Simulation-to-real transfer learning capabilities:  

- Domain Randomization + PPO
- Domain Adaptation + PPO

The trainings are carried out in the Duckietown Gym environment. To test the methods in the real life environment, one can acquire the [Duckiebot hardware](https://www.duckietown.org/about/hardware)  or submit the trained model to the [AI Driving Olympics Challenge server](https://challenges.duckietown.org/v4/).

***

## Outline

[1. Setup Environment](##-1.-Setup-Environment)

[2. Manual Control example](##-2.-Manual-Control-example)

[3. Tutorial 1: Training with Domain Randomization](#tutorial_dr)

[4. Tutorial 2: Training with Domain Adaptation](#tutorial_da)

[5. Testing in the Duckietown Gym Simulator](#testing_agent)

***

### Prerequisites

This tutorial currently works on Linux OS.
Anaconda and Python 3.6 is required.
***
<p align="center" >
<img src="art/concept.png" alt="Concept" width="300">
</p>

***

## 1. Setup Environment

This repository provides both # conda # and # docker # ways to set up a working environment for the tutorial. Choose yours wisely.

### 1.1 Conda environment

- Run conda environment setup:

```$ bash setup_conda_environment.sh```

- Run jupyter notebook:

```$ xvfb-run -a -s "-screen 0 1400x900x24" jupyter notebook  --ip 0.0.0.0 --port <portnumber> --no-browser --allow-root```

or

- Run jupyter lab:

```$ xvfb-run -a -s "-screen 0 1400x900x24" jupyter lab  --ip 0.0.0.0 --port <portnumber> --no-browser --allow-root```

- Access your editor with a browser ```http://localhost: \<portnumber\>```

### 1.2 Docker environment

- Build the docker environment

```$ docker build . --tag sim2real_image```

- Run the docker environment

```$ nvidia-docker run --name sim2real_image -v $(pwd):/selfdriving_with_sim2real --shm-size=2gb -t -d sim2real_image:latest bash```

*Note:* for development with the docker container  we recommend using Visual Studio Code with this setup (<https://code.visualstudio.com/docs/remote/containers>).

***

## 2. Manual Control example
  
- Change directory:

```$ cd gym-duckietown```

- Run the following code:

```$ python3 manual_control.py --env-name Duckietown-udem1-v0 --map-name loop_dyn_duckiebots --domain-rand --distortion```

At this point you will be able to run the Duckietown Gym environment and manual control the Duckiebot.

***

## 3. Tutorial 1: Training with Domain Randomization
  
  Open **01. Training with Domain Randomization.ipynb** and follow instructions.
  During the training the logs arre saved in the *artifacts/* directory.
  
<p align="center" >
<img src="art/just_policy.png" alt="Concept" width="300">
</p>
# tutorial_dr

***

## 4. Tutorial 2: Training with Domain Adaptation

### 4.1 Data generation

#### Genrate data from simulator

- Change directory

```$ cd domain_adaptation```

- Run the following script. This might take a while...

```$ python generate_dataset_sim.py --rollouts 200 --seq_len 10 --data_dir /selfdriving_with_sim2real/data/train/sim```

This will generate 8.000 samples.

### Generate data from real

- Remain in the  representation_learning  directory.
- Run the following script. This will take less...

```$ python generate_dataset_real.py```

### 4.2 Training the Domain Adaptation network (UNIT network)

Change directory

```$ cd /selfdriving_with_sim2real/representation_learning/unit/```

Run the following code

```$ CUDA_VISIBLE_DEVICES=1 python train.py  --exp_name training_unit```

This code will generate data into the **data** directory.

### 4.3 Training the agent with the Domain Adaptation network

- Open **01. Training with Domain Adaptation.ipynb** and follow instructions.
  
***

## 5. Testing in the Duckietown Gym Simulator
  
- Open **03. Testing the agent.ipynb** and follow instructions.

### References
