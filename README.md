# baby env
BabyEnv is a small OpenAI Gym environment to simulate scheduling problem as an MDP problem.
This is simplified 


## Getting started

First install baby_env:  
```sh
virtualenv ~/.virtualenvs/baby/ -p python3
source ~/.virtualenvs/baby/bin/activate

make install # or pip install -e .
```

## Run/assess

To launch a training phase with A2C algorithm:  
```sh
python baby/rl/a2c_xp.py
```

To evaluate model:  
```sh
python baby/rl/a2c_assess.py
```

