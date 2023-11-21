import numpy as np
import random
import math
import gymnasium
from gymnasium import spaces

class Double_Component(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self):

        # Action space: (0) no action, (1) inspection
        self.action_space = spaces.Discrete(2)

        # Observation space:
        # | Num | 0                                                    | 1                                                 |
        # |-----|------------------------------------------------------|---------------------------------------------------|
        # | 0   | System downtime overflow (hours)                     | Time since inspection (weeks)                     |
        # | 1   | (Blade) Number of repairs since last replacement     | (Blade) Time since maintenance action (weeks)     |
        # | 2   | (Generator) Number of repairs since last replacement | (Generator) Time since maintenance action (weeks) |
        COMPONENTS = 2
        low = np.zeros((COMPONENTS+1,2),dtype=np.float32)
        high = np.ones((COMPONENTS+1,2),dtype=np.float32)
        high *= 10**6
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.system_downtime_overflow = None
        self.time_since_inspection = None
        self.number_of_repairs_since_replacement = None
        self.time_since_maintenance = None


    def step(self, action):
        COMPONENTS = 2
        DISCOUNT_FACTOR = 0.04
        POWER_CAPACITY = 5
        CAPACITY_FACTOR = 0.49
        VALUE_OF_MWH = 40
        INSPECTION_COST = 2000
        CONVERSION_RATE = 1.14

        COST_OF_MAINTENANCE = np.array([    # Cost of repair, cost of replacement
            [4000, 200000],
            [50000, 150000]], dtype=np.float32)
        COST_OF_MAINTENANCE *= CONVERSION_RATE

        DURATION_OF_MAINTENANCE = np.array([    # Duration of repair, duration of replacement
            [3, 70],
            [10, 50]], dtype=np.float32)

        SCALE_PARAM = np.array([    # Weibull parameters for the markov transition to degradated, critical, failure states
            [23.02, 2.88, 2.88],
            [15.38, 1.92, 1.92]], dtype=np.float32)
        SCALE_PARAM = 1/(SCALE_PARAM*52)
        SHAPE_PARAM = np.array([
            [1.2, 1.2, 1.2],
            [1.2, 1.2, 1.2]], dtype=np.float32)
        for component in range(0, COMPONENTS):
            SHAPE_PARAM[component,0:] *= (1.1**self.number_of_repairs_since_replacement[component])
        pr_normal = np.zeros(COMPONENTS, dtype=np.float32)
        pr_degraded = np.zeros(COMPONENTS, dtype=np.float32)
        pr_critical = np.zeros(COMPONENTS, dtype=np.float32)
        pr_failed = np.zeros(COMPONENTS, dtype=np.float32)

        reward = 0
        done = False

        if self.system_downtime_overflow >= 7*24:
            self.system_downtime_overflow -= 7*24
            if action == 1:
                reward -= INSPECTION_COST
        else:
            production_hours = 7*24
            if self.system_downtime_overflow > 0:
                production_hours -= self.system_downtime_overflow
                self.system_downtime_overflow = 0

            reliability = [1] * COMPONENTS
            system_reliability = 1
            for component in range (0, COMPONENTS):
                pr_normal[component] = math.exp((SCALE_PARAM[component,0]**SHAPE_PARAM[component,0])*((((self.time_since_maintenance[component]\
                -self.time_since_inspection))**SHAPE_PARAM[component,0])-((self.time_since_maintenance[component]))**SHAPE_PARAM[component,0]))

                pr_degraded[component] = (1-pr_normal[component])*math.exp((SCALE_PARAM[component,1]**SHAPE_PARAM[component,1])*((((self.time_since_maintenance[component]\
                -self.time_since_inspection))**SHAPE_PARAM[component,1])-((self.time_since_maintenance[component]))**SHAPE_PARAM[component,1]))

                pr_critical[component] = (1-pr_normal[component]-pr_degraded[component])*math.exp((SCALE_PARAM[component,2]**SHAPE_PARAM[component,2])*((((self.time_since_maintenance[component]\
                -self.time_since_inspection))**SHAPE_PARAM[component,2])-((self.time_since_maintenance[component]))**SHAPE_PARAM[component,2]))

                pr_failed[component] = 1 - pr_normal[component] - pr_degraded[component] - pr_critical[component]
                reliability[component] = 1 - pr_failed[component]
                system_reliability *= reliability[component]
                self.time_since_maintenance[component] += 1
            
            if action == 0:
                self.time_since_inspection += 1
            elif action == 1:
                self.time_since_inspection = 0
                reward -= INSPECTION_COST
                production_hours -= 1
                for component in range (0, COMPONENTS):
                    random_number = random.randint(0,10**5)/10**5
                    if pr_failed[component] >= random_number:   # System component has failed
                        reward -= COST_OF_MAINTENANCE[component,1]
                        self.time_since_maintenance[component] = 0
                        production_hours -= DURATION_OF_MAINTENANCE[component,1]
                    elif (pr_failed[component] + pr_critical[component]) >= random_number:  # System component is in critical state
                        reward -= COST_OF_MAINTENANCE[component,1]
                        self.time_since_maintenance[component] = 0
                        production_hours -= DURATION_OF_MAINTENANCE[component,1]
                    elif (pr_failed[component] + pr_critical[component] + pr_degraded[component]) >= random_number: # System component is in degraded state
                        reward -= COST_OF_MAINTENANCE[component,0]
                        self.time_since_maintenance[component] = 0
                        production_hours -= DURATION_OF_MAINTENANCE[component,0] 

            if production_hours > 0:
                reward += system_reliability * production_hours * POWER_CAPACITY * CAPACITY_FACTOR * VALUE_OF_MWH
            else:
                self.system_downtime_overflow = - production_hours

        reward /= (1 + DISCOUNT_FACTOR)**(self.runtime/52)
        reward /= 1000000

        self.runtime += 1
        if self.runtime >= 3000000:
            done=True
        truncated = False
        info = {}
        obs = np.zeros((COMPONENTS+1, 2),dtype=np.float32)
        obs[0,0] = self.system_downtime_overflow
        obs[0,1] = self.time_since_inspection
        for component in range(0, COMPONENTS):
            obs[component+1, 0] = self.number_of_repairs_since_replacement[component]
            obs[component+1, 1] = self.time_since_maintenance[component]
        return np.array(obs, dtype=np.float32), reward, done, truncated, info

    def render(self):
        pass

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        COMPONENTS = 2
        MAX = [2*52, 2*52]
        self.runtime = 0
        self.system_downtime_overflow = 0
        self.time_since_inspection = 0
        self.number_of_repairs_since_replacement = np.zeros(COMPONENTS,dtype=np.float32)
        self.time_since_maintenance = np.zeros(COMPONENTS,dtype=np.float32)

        obs = np.zeros((COMPONENTS+1, 2),dtype=np.float32)
        obs[0,0] = self.system_downtime_overflow
        obs[0,1] = self.time_since_inspection
        for component in range(0, COMPONENTS):
            if options == None:
                self.time_since_maintenance[component] = random.randint(0, MAX[component])

            obs[component+1, 0] = self.number_of_repairs_since_replacement[component]
            obs[component+1, 1] = self.time_since_maintenance[component]
        info = {}
        return np.array(obs, dtype=np.float32), info