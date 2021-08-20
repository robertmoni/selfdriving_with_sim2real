# Self Driving Agent training with Simulation-to-Real transfer learning methods

![Tools](art/tools.png)

This repository contains a brief tutorial on training a Reinforcement Learning agent for Lane Following in the Duckietown environment.
The agent is trained using the Proximal Policy Optimization (PPO) algorithm with Generalized Advantage Estimation (GAE) method in an Actor-Critic framework.
Since the Duckietown environment supports deployment into real life environment with Duckiebot robots, we provide our methods with Simulation-to-real transfer learning capabilities:  

- Domain Randomization + PPO
- Domain Adaptation + PPO

<p align="center" >
<img src="art/concept.png" alt="Concept" width="300">
</p>
The trainings are carried out in the Duckietown Gym environment. To test the methods in the real life environment, one can acquire the [Duckiebot hardware](https://www.duckietown.org/about/hardware)  or submit the trained model to the [AI Driving Olympics Challenge server](https://challenges.duckietown.org/v4/).

***

## Outline

[1. Setup Environment](##-1.-Setup-Environment)

[2. Tutorial 1: Training agent with PPO and Domain Randomization](#tutorial_dr)

[3. Tutorial 2: Training with PPO and Domain Adaptation](#tutorial_da)

[4. Testing in the Duckietown Gym Simulator](#testing_agent)

<p align="center" >
<img src="art/sim_trained.gif" alt="Concept" width="300">
<img src="art/real_trained.gif" alt="Concept" width="300">
</p>

***

### Prerequisites

This tutorial currently works using Docker environment.
***

***

## 1. Setup Environment

This repository provides a # docker # image to set up a working environment for the tutorial.

- Build the docker environment

```$ docker build . --tag sim2real_image```

- Run the docker environment

```$ nvidia-docker run --name sim2real_container -v $(pwd):/selfdriving_with_sim2real -p 7090:6006 -p 7091:8888 --shm-size=4gb -d sim2real_image:latest bash```

- or, if you don't have any gpu:

```$ docker run --name sim2real_container -v $(pwd):/selfdriving_with_sim2real -p 7090:6006 -p 7091:8888  --shm-size=4gb -d sim2real_image:latest bash```

- Attach to docker (if not attached by default)

```docker attach sim2real_container```

**From this point, all code will be ran inside the docker container!!!**

- Run Jupyter Lab
```xvfb-run -a -s "-screen 0 1400x900x24" jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root```

*Note1: since the docker container act as a headles display, we need to set up a virtual display with **xvfb**.*

- Access Jupyter Lab in your browser with the attached port number (**7091**)

[http://localhost:7091](http://localhost:7091)

*Note2: if the docker is running on a remote machine, change the **localhost** part to the ipaddress of the remote machine.*

*Note3: for development with the docker container  we recommend using Visual Studio Code with this setup (<https://code.visualstudio.com/docs/remote/containers>).*

***

## 2. Tutorial 1: Training agent with PPO and Domain Randomization
  
  Open **01. Training with Domain Randomization.ipynb** and follow instructions.
  During the training the logs arre saved in the *artifacts/* directory.
  
<p align="center" >
<img src="art/just_policy.png" alt="Concept" width="300">
</p>

***

## 3. Tutorial 2: Training with PPO and Domain Adaptation

### 3.1 Data generation

#### Generate data from simulator

- Change directory

```$ cd domain_adaptation```

- Run the following script. This might take a while...

```$ python generate_dataset_sim.py --rollouts 200 --seq_len 10 --data_dir /selfdriving_with_sim2real/data/train/sim```

This will generate 8000 samples.

### Generate data from real

- Remain in the  representation_learning  directory.
- Run the following script. This will take less...

```$ python generate_dataset_real.py```

### 3.2 Training the Domain Adaptation network (UNIT network)

Change directory

```$ cd /selfdriving_with_sim2real/representation_learning/unit/```

Run the following code

```$ CUDA_VISIBLE_DEVICES=1 python train.py  --exp_name training_unit```

This code will generate data into the **data** directory.

### 3.3 Training the agent with the Domain Adaptation network

- Open **02. Training with Domain Adaptation.ipynb** and follow instructions.
  
***

## 4. Testing in the Duckietown Gym Simulator
  
- Open **03. Testing the agent.ipynb** and follow instructions.

### References
