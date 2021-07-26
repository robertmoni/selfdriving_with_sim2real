# Self Driving Agent training with Simulation-to-Real trasnfer using Python, Ray RLlib and gym-duckietown
![](art/tools.png)

This repository contains the steps of a tutorial to succesfully train a Reinforcement Learning agent with Deep Reainforcement Learning. Furthermore, to tackle the simulation-to-real transfer two different methods are presented: Domain Randomization and Domain Adaptation. The trainings are carried out in the Duckietown Gym environment. 


***
# Outline 
[1. Setup Environment](#setup_environment)


[2. Manaul Control example](#manual_control)


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
# 1. Setup Environment
This repository provides both # conda # and # docker # ways to set up a working environmnet for the tutorial. Choose your's wisely. 
## 1.1 Conda environment


Run conda environment setup:
```$ bash setup_conda_environment.sh```

Run jupyter notebook:
```$ xvfb-run -a -s "-screen 0 1400x900x24" jupyter notebook  --ip 0.0.0.0 --port <portnumber> --no-browser --allow-root  ```

or

Run jupyter lab:
```$ xvfb-run -a -s "-screen 0 1400x900x24" jupyter lab  --ip 0.0.0.0 --port <portnumber> --no-browser --allow-root  ```

* Access your editor with a browser http://localhost:<portnumber>

## 1.2 Docker environment

Build the docker environmnet
```$ docker build . --tag sim2real_image```


Run the docker environent

```$ nvidia-docker run --name sim2real_image -v $(pwd):/selfdriving_with_sim2real --shm-size=2gb -t -d sim2real_image:latest bash```


*Note:* for developement with the docker contaainer  we reccomand using Visual Studio Code with this setup (https://code.visualstudio.com/docs/remote/containers).
# 2. Manual Control example
  
Run the follwong code:

```cd gym-duckietown```
```python3 manual_control.py --env-name Duckietown-udem1-v0 --map-name loop_dyn_duckiebots --domain-rand --distortion```


At this point you will be able to run the Duckietown Gym environment and manual control the Duckiebot.


***
# 3. Tutorial 1: Training with Domain Randomization
  
  
  Open **01. Training tutorial.ipynb** and follow instructions.
  During the training the logs arre saved in the *artifacts/* directory.
  
  


<p align="center" >
<img src="art/just_policy.png" alt="Concept" width="300">
</p>
#tutorial_dr

***
# 4. Tutorial 2: Training with Domain Adaptation

## 4.1 Data generation

### Genrate data from simulator
-  Change direcorty 

```$ cd representation_learning```

- Run the following script. This might take a while...

```$ python generate_dataset_sim.py --rollouts 400 --seq_len 10 --data_dir /selfdriving_with_sim2real/data/train/sim ```

This will generate 16.000 samples.

### Genrate data from real

- Remain in the <bd> representation_learning </bd> directory.
- Run the following script. This will take less...
```python generate_dataset_real.py ```






**TBD**
  
  
#tutorial_da

***
# 5. Testing in the Duckietown Gym Simulator
  
**TBD**
  
#testing_agent

