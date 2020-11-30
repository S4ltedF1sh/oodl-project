from make_env import make_env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.ppo import MlpPolicy

i = 11
landmarks = 2
foods = 5
num = 15
dis = 2

env = make_env("deep_learning_scenario", num_landmarks=landmarks, num_foods=foods, seed=i)
env = make_vec_env(lambda: env, n_envs=4)
#model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard/normal/ppo_{}o{}f/".format(landmarks, foods), learning_rate=3e-5)
model = PPO.load("saved_models/ppo_{}o{}f/PPO_{}mil".format(landmarks, foods, i*dis-dis), env)

model.learn(total_timesteps=dis*1000000)
model.save("saved_models/ppo_{}o{}f/PPO_{}mil".format(landmarks, foods, i*dis))
for j in range (i+1, i+num):
    model.learn(total_timesteps=dis*1000000, reset_num_timesteps=False)
    model.save("saved_models/ppo_{}o{}f/PPO_{}mil".format(landmarks, foods, j*dis))