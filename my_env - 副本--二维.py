import numpy as np
import pandas as pd

class Env:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transDelay = 3
        self.cost_deployment = 5
        self.observation_space = dict(
            {
                "taxi": np.empty(shape=(self.x, self.y), dtype=np.uint8),##environment
               "bus": np.empty(shape=(self.x, self.y), dtype=np.uint8)##environment

            }
        )

        self.action_space = [0, 1] ##action[0 1]
        self.obs_taxi = self.observation_space["taxi"]
        self.obs_bus = self.observation_space["bus"]
    def reset(self):
        self.obs_taxi = np.random.randint(low=0,high=8,size=(5,5))
        self.obs_bus = np.random.randint(low=0,high=2,size=(5,5))
        return self._get_obs()
    def _get_obs(self):
        return {"taxi":self.obs_taxi, "bus": self.obs_bus}
    def _get_cost(self,obs_offloadtoEN):
        computingDelay_EN = np.sum(obs_offloadtoEN) * 5
        computingDelay_Bus = np.sum(self.obs_bus) * 6
        cost_offtoEN = computingDelay_EN + self.transDelay
        cost_offtoBus = computingDelay_Bus + self.transDelay
        return -(cost_offtoEN + cost_offtoBus)
    def _get_reward(self,obs_offloadtoEN):
        reward = -np.sum(self.obs_bus == 1)
        reward += self._get_cost(obs_offloadtoEN)
        return reward
    def _get_done(self):
        return 0
    def step(self, action):  #TODO
        self.obs_taxi = self.observation_space["taxi"]
        self.obs_bus = self.observation_space["bus"]
        if action:
            obs_offloadtoEN = self.obs_taxi - self.obs_bus
            obs_offloadtoEN = np.maximum(obs_offloadtoEN, 0)
        else:
            obs_offloadtoEN = self.obs_taxi
        self.observation_space = self._get_obs()
        reward = self._get_reward(obs_offloadtoEN)
        print(reward)
        done = self._get_done()
        info = None #TODO
        return self.observation_space, reward, done, info



