import gym
from collections import deque
import numpy as np
import cv2
import torch
"""

Args:
    env: gym environment
    action: [v, omega] action of the agent
returns:
        (frame_stack, h, w) state
"""
def get_skipped_stacked_frames(env, action):


    return


class duckieEnvWrapper:
    def __init__(self, env, frame_stack=4, frame_skip=4):
        self.env = env
        self.action_space = env.action_space.shape  # (2, )  low = -1, high = 1
        self.observation_space = (4, ) + (84, 84)  # (frame_stack, downsampled_h, downsampled_w)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip

        self.stacked_obs = deque(maxlen=frame_stack)

    def grayscale(self, obs):
        """
        Convert the observation to grayscale
        """
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # dim (h, w)
    
    def downsample(self, obs):
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)  # downsample to 84x84 like atari

    def reset(self):
        obs = self.env.reset()  # dim (h, w, c)
        obs = self.grayscale(obs)  # dim (h, w)
        obs = self.downsample(obs)  # dim (h, w)
        self.stacked_obs.clear()
        for _ in range(self.frame_stack):  # fill the deque
            self.stacked_obs.append(obs)
        return np.stack(self.stacked_obs, axis=0)  # (frame_stack, h, w, c)

    """
    We sum the rewards of the skipped frames? yes, based on this discussion http://disq.us/p/1syf8i8
    """
    def get_skipped_frame(self, action):
        """
        Get the skipped frame from the environment
        """
        obs = None
        total_reward = 0

        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        obs = self.grayscale(obs)  # dim (h, w)
        obs = self.downsample(obs)  # dim (84 x 84)
        return obs, total_reward,  done
    
    """
    action = [0, 1] # [v, omega]
    
    """
    def step(self, action):
        obs, reward, done = self.get_skipped_frame(action)
        self.stacked_obs.append(obs)
        state = np.stack(self.stacked_obs, axis=0)
        return state, reward, done



def evaluate(make_env, map_name, agent, num_episodes=10, velocity=0.2):
    print(f'evaluating....')
    env = make_env(map_name)
    env = duckieEnvWrapper(env)
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                omega = agent.get_action_and_value(obs)[0]
            env_action = [velocity, omega.cpu().numpy()]
            obs, reward, done = env.step(env_action)
            total_reward += reward
    return total_reward / num_episodes
