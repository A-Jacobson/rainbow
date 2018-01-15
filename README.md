# Rainbow
Rainbow: Combining Improvements in Deep Reinforcement Learning

## Setup

1. Follow setup instructions at https://github.com/openai/gym to install openai gym and atari games.
```
pip install gym
pip install -e '.[atari]'
```

2. install pytorch and torchvision:
```
conda install pytorch torchvision -c pytorch
```

3. install tensorflow and tensorboardX for logging.
```
pip install tensorboard
pip install tensorboardX
```

## Contents
- [DQN tutorial](https://github.com/A-Jacobson/rainbow/blob/master/Minimal_DQN.ipynb)
- Vanilla DQN agent class

## Usage
To train an agent on breakout for 2000000 steps.
```
python train.py --env Breakout-v0 --num_steps 2000000 --agent_name breakoutdqn
```

To watch your agent play 20 episodes of breakout.
```
python play.py --env Breakout-v0 --num_episodes 20 --agent_name breakoutdqn
```

