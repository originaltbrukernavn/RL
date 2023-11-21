import gymnasium
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

gymnasium.register(
     id='Double_component-v0',
     entry_point='1 Double component environment:Double_Component',
     max_episode_steps=10000
    #  kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)

envs = make_vec_env(env_id='Double_component-v0', seed=1, n_envs=5)

RL_Path = os.path.join(r'C:\Users\PC\Desktop\Reinforcement Learning\Windmill problem\Double Component\Training', 'RL_Double')
Log_Path = os.path.join(r'C:\Users\PC\Desktop\Reinforcement Learning\Windmill problem\Double Component\Training', 'Logs')

model = PPO('MlpPolicy', envs, verbose=0, tensorboard_log=Log_Path)
model.learn(total_timesteps=100000, log_interval=5, progress_bar=True)
model.save(RL_Path)
envs.close