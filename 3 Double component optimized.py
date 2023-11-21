import gymnasium
import os
import math
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

gymnasium.register(
     id='Double_component-v0',
     entry_point='1 Double component environment:Double_Component',
     max_episode_steps=10000
    #  kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
)
env = gymnasium.make('Double_component-v0')
RL_Path = os.path.join(r'C:\Users\PC\Desktop\Reinforcement Learning\Windmill problem\Double Component\Training', 'RL_Double')
Log_Path = os.path.join(r'C:\Users\PC\Desktop\Reinforcement Learning\Windmill problem\Double Component\Training', 'Logs')
model = PPO.load(RL_Path, env=env)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

time = 0
COMPONENTS = 2
SCALE_PARAM = np.array([
    [23.02, 2.88, 2.88],
    [15.38, 1.92, 1.92]], dtype=np.float32)
SCALE_PARAM = 1/(SCALE_PARAM*52)
SHAPE_PARAM = np.array([
    [1.2, 1.2, 1.2],
    [1.2, 1.2, 1.2]], dtype=np.float32)

obs, _ = env.reset(options=1)
done = False
truncated = False
score = 0
sum_action = 0
reliability = np.ones(COMPONENTS, dtype=np.float32)
time_since_maintenance = np.zeros(COMPONENTS, dtype=np.float32)
r = np.ones(COMPONENTS, dtype=np.float32)
number_of_repairs_since_replacement = np.zeros(COMPONENTS, dtype=np.float32)
pr_normal = np.zeros(COMPONENTS, dtype=np.float32)
pr_degraded = np.zeros(COMPONENTS, dtype=np.float32)
pr_critical = np.zeros(COMPONENTS, dtype=np.float32)
pr_failed = np.zeros(COMPONENTS, dtype=np.float32)

x_axis = []
reliability1 = []
reliability2 = []
reliability3 = []
fail1 = []
fail2 = []
crit1 = []
crit2 = []
deg1 = []
deg2 = []

t_i = []
i = []
r1 = []
r2 = []
r3 = []
f1 = []
f2 = []
c1 = []
c2 = []
d1 = []
d2 = []
while not done:
    time+=1
    action, state = model.predict(obs)
    sum_action += action
    obs, reward, done, truncated, info = env.step(action)
    system_downtime_overflow = obs[0,0]
    time_since_inspection = obs[0,1]
    for component in range(0, COMPONENTS):
        number_of_repairs_since_replacement[component] = obs[component+1, 0]
        time_since_maintenance[component] = obs[component+1, 1]

    SHAPE_PARAM = np.array([
        [1.2, 1.2, 1.2],
        [1.2, 1.2, 1.2]], dtype=np.float32)
    for component in range(0, COMPONENTS):
        SHAPE_PARAM[component,0:] *= (1.1**number_of_repairs_since_replacement[component])
    system_reliability = 1
    r_sys = 1
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
        r[component] = 1
        r_sys *= r[component]
    score += reward
    if time>=52*2:
        done=True
    fail1.append(pr_failed[0])
    fail2.append(pr_failed[1])
    crit1.append(pr_critical[0]+pr_failed[0])
    crit2.append(pr_critical[1]+pr_failed[1])
    deg1.append(pr_degraded[0]+pr_critical[0]+pr_failed[0])
    deg2.append(pr_degraded[1]+pr_critical[1]+pr_failed[1])
    x_axis.append(time/52)
    reliability1.append(reliability[0])
    reliability2.append(reliability[1])
    reliability3.append(system_reliability)
    r1.append(r[0])
    r2.append(r[1])
    r3.append(r_sys)
    t_i.append(time_since_inspection)
    i.append(sum_action)

print('Years: {} - Inspections: {} - Score: {}'.format(time/52, sum_action, score))
env.close
# figure, axis = plt.subplots(2, 2)
# axis[0, 0].plot(x_axis, reliability1, color = 'r')
# axis[0, 0].plot(x_axis, r1, color = 'b') 
# axis[0, 0].set_title("Reliability component 1") 

# axis[0, 1].plot(x_axis, reliability2, color = 'r')
# axis[0, 1].plot(x_axis, r2, color = 'b') 
# axis[0, 1].set_title("Reliability component 2") 

# axis[1, 0].plot(x_axis, reliability3, color = 'r')
# axis[1, 0].plot(x_axis, r3, color = 'b') 
# axis[1, 0].set_title("System reliability") 

# axis[1, 1].plot(i, t_i) 
# axis[1, 1].set_title("Time since inspection")
# plt.show()

figure, axis = plt.subplots(1, 2)
axis[0].plot(x_axis, fail1, color = 'r')
axis[0].plot(x_axis, crit1, color = 'b') 
axis[0].plot(x_axis, deg1, color = 'c')
axis[0].set_title("Reliability component 1")

axis[1].plot(x_axis, fail2, color = 'r')
axis[1].plot(x_axis, crit2, color = 'b') 
axis[1].plot(x_axis, deg2, color = 'c')
axis[1].set_title("Reliability component 2") 
plt.show()