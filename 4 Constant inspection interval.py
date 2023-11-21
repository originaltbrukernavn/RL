import numpy as np
import math
import random
import matplotlib.pyplot as plt

TIME_FRAME = 52*100
LOOPS = 1
MIN_INSPECTION_INTERVAL = 1
MAX_INSPECTION_INTERVAL = 200
SPAN = MAX_INSPECTION_INTERVAL - MIN_INSPECTION_INTERVAL + 1
sum_reward = np.zeros(SPAN, dtype=np.float32)

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

for loop in range(0,LOOPS):
    print('Loop: {}'.format(loop+1))
    for inspection_interval in range (MIN_INSPECTION_INTERVAL, MAX_INSPECTION_INTERVAL+1):
        number_of_repairs_since_replacement = np.zeros(COMPONENTS,dtype=np.float32)
        system_downtime_overflow = 0
        time_since_inspection = 0
        time_since_maintenance = np.zeros(COMPONENTS,dtype=np.float32)
        for runtime in range (1,TIME_FRAME+1):
            if (runtime % inspection_interval) == 0:
                action = 1
            else:
                action = 0
            SHAPE_PARAM = np.array([
                [1.2, 1.2, 1.2],
                [1.2, 1.2, 1.2]], dtype=np.float32)
            for component in range(0, COMPONENTS):
                SHAPE_PARAM[component,0:] *= (1.1**number_of_repairs_since_replacement[component])
            pr_normal = np.zeros(COMPONENTS, dtype=np.float32)
            pr_degraded = np.zeros(COMPONENTS, dtype=np.float32)
            pr_critical = np.zeros(COMPONENTS, dtype=np.float32)
            pr_failed = np.zeros(COMPONENTS, dtype=np.float32)

            reward = 0
            done = False

            if system_downtime_overflow >= 7*24:
                system_downtime_overflow -= 7*24
                if action == 1:
                    reward -= INSPECTION_COST
            else:
                production_hours = 7*24
                if system_downtime_overflow > 0:
                    production_hours -= system_downtime_overflow
                    system_downtime_overflow = 0

                reliability = [1] * COMPONENTS
                system_reliability = 1
                for component in range (0, COMPONENTS):
                    pr_normal[component] = math.exp((SCALE_PARAM[component,0]**SHAPE_PARAM[component,0])*((((time_since_maintenance[component]\
                    -time_since_inspection))**SHAPE_PARAM[component,0])-((time_since_maintenance[component]))**SHAPE_PARAM[component,0]))

                    pr_degraded[component] = (1-pr_normal[component])*math.exp((SCALE_PARAM[component,1]**SHAPE_PARAM[component,1])*((((time_since_maintenance[component]\
                    -time_since_inspection))**SHAPE_PARAM[component,1])-((time_since_maintenance[component]))**SHAPE_PARAM[component,1]))

                    pr_critical[component] = (1-pr_normal[component]-pr_degraded[component])*math.exp((SCALE_PARAM[component,2]**SHAPE_PARAM[component,2])*((((time_since_maintenance[component]\
                    -time_since_inspection))**SHAPE_PARAM[component,2])-((time_since_maintenance[component]))**SHAPE_PARAM[component,2]))

                    pr_failed[component] = 1 - pr_normal[component] - pr_degraded[component] - pr_critical[component]
                    reliability[component] = 1 - pr_failed[component]
                    system_reliability *= reliability[component]
                    time_since_maintenance[component] += 1
                
                if action == 0:
                    time_since_inspection += 1
                elif action == 1:
                    time_since_inspection = 0
                    reward -= INSPECTION_COST
                    production_hours -= 1
                    for component in range (0, COMPONENTS):
                        random_number = random.randint(0,10**5)/10**5
                        if pr_failed[component] >= random_number:   # System component has failed
                            reward -= COST_OF_MAINTENANCE[component,1]
                            time_since_maintenance[component] = 0
                            production_hours -= DURATION_OF_MAINTENANCE[component,1]
                        elif (pr_failed[component] + pr_critical[component]) >= random_number:  # System component is in critical state
                            reward -= COST_OF_MAINTENANCE[component,1]
                            time_since_maintenance[component] = 0
                            production_hours -= DURATION_OF_MAINTENANCE[component,1]
                        elif (pr_failed[component] + pr_critical[component] + pr_degraded[component]) >= random_number: # System component is in degraded state
                            reward -= COST_OF_MAINTENANCE[component,0]
                            time_since_maintenance[component] = 0
                            production_hours -= DURATION_OF_MAINTENANCE[component,0] 

                if production_hours > 0:
                    reward += system_reliability * production_hours * POWER_CAPACITY * CAPACITY_FACTOR * VALUE_OF_MWH
                else:
                    system_downtime_overflow = - production_hours

            reward /= (1 + DISCOUNT_FACTOR)**(runtime/52)
            reward /= 1000000

            sum_reward[inspection_interval-MIN_INSPECTION_INTERVAL] += reward
sum_reward /= LOOPS
i = []
rw = []
optimal_inspection_interval = 0
rw_max = 0
for inspection_interval in range (MIN_INSPECTION_INTERVAL, MAX_INSPECTION_INTERVAL+1):
    if sum_reward[inspection_interval-MIN_INSPECTION_INTERVAL] > rw_max:
        rw_max = inspection_interval-MIN_INSPECTION_INTERVAL
        optimal_inspection_interval = inspection_interval
    i.append(inspection_interval)
    rw.append(sum_reward[inspection_interval-MIN_INSPECTION_INTERVAL])

print('Optimal inspection interval: {}. Reward: {}'.format(optimal_inspection_interval, rw_max))
plt.plot(i, rw)
plt.show()