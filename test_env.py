from make_env import make_env
import random
import numpy as np
import time
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.ppo import MlpPolicy

#i = 1
landmarks = 0
foods = 5
env = make_env("deep_learning_scenario", num_landmarks=landmarks, num_foods=foods)

model = PPO.load("saved_models/ppo_{}o{}f/PPO_{}mil".format(landmarks, foods, 4), env)
obs = env.reset()
finish = False
i = 0
total_reward = 0
while not finish:
    #action = model.predict(obs)
    #print(action)
    action = model.predict(obs)
    obs, reward, done, info = env.step(action[0])
    finish = done
    i = i + 1
    total_reward = total_reward + reward
    print(i, obs)
    print(total_reward)
    env.render()
    #time.sleep(0.1)
