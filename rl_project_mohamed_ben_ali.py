# -*- coding: utf-8 -*-
"""RL_Project_Mohamed BEN ALI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1k0I8rdZhrCL_v_BgbS5GdL7Fg648OWFM

# Reinforcement Learning Project
## Article: *Bridging the gap between regret minimization and best arm identification, with application to A/B tests*
### Student: Mohamed BEN ALI

In this notebook, we will attempt to implement some of the algorithms presented in the article, and reproduce some of the experiments.

## Framework and setup

In this part we import the relevant functions, and present the classes we developed for the bandit model.
"""

import numpy as np
import matplotlib.pyplot as plt
from framework import *

"""### Environment
#### Pulling arms

The `Arm` class is just a convenience to draw samples from a certain distribution. Its usage is shown thereafter.
"""

mean = 0
std = 1

test_arm = Arm(mean, std, gaussian_sampling)
results = test_arm.pull(times=10000)
print(f"Empirical mean = {results.mean()} (theoretical = {mean})")
print(f"Empirical std = {results.std()} (theoretical = {std})")

"""#### Creating an environment

An `Environment` is an object defined by a list of arms. It lets a user pull an arm and exposes the history of observed rewards.
"""

# Create an environement
test_arm_bis = Arm(mean, std, gaussian_sampling)
test_env = Environment([test_arm, test_arm_bis])

# Pull an arm
test_env.pull_arm(1)

# Display the reward history
test_env.reward_history

# Reset the reward history
test_env.reset_history()
test_env.reward_history

"""### Test with an Explore-Then-Commit (ETC) agent

We implement an ETC agent such as described in the article (*cf*. page 2). We then compare the theoretical regret to the empirical regret.
"""

def run_experiment(agent, confidence, std, mean1, mean2, n_steps, n_experiments):
    # Create an environment with 2 arms
    arm1 = Arm(mean1, std, gaussian_sampling)
    arm2 = Arm(mean2, std, gaussian_sampling)
    env = Environment([arm1, arm2])

    regrets = []
    decision_times = []

    for _  in range(n_experiments):
        results = agent.play(n_steps, confidence, env)
        regrets.append(results.decision_regret)
        decision_times.append(results.decision_time)

    return np.mean(regrets), np.mean(decision_times)

"""We compute the regret for a large number of experiments in order to increase the precision of empirical results."""

agent = ETC_Agent()
confidence = 0.02
std = 1
mean1 = 0
mean2 = 1
n_steps = 100
n_experiments = 1000

regret, decision_time = run_experiment(agent, confidence, std, mean1, mean2, n_steps, n_experiments)

print(f"Average regret at time of decision: {np.mean(regret)}, Average decision time: {np.mean(decision_time)}.")
theoretical_regret = (8*std**2)/(mean2-mean1)*np.log(1/confidence)
print(f"Theoretical regret bound: {theoretical_regret}")

"""According to the article, the regret at the time of decision is bound by a number slightly larger than
$\frac{8\sigma^2}{\Delta}\log(1/\delta)$
, which seems to be respected in the experiment above.

## The UCB $_{\alpha}$ algorithm

In this section, we implement and experiment with the $UCB_{\alpha}$ algorithm proposed in the article.
"""

# Coefficient needed to compute the theoretical regret
def c(alpha):
    if alpha == 1 : return 1
    else : return np.min([(alpha+1)**2 / 4, 4*alpha**2 / (1 - alpha)**2])

alpha = 1
agent = UCBAlpha_Agent(alpha)
confidence = 0.02
std = 1
mean1 = 0
mean2 = 1
n_steps = 1000
n_experiments = 100

regret, decision_time = run_experiment(agent, confidence, std, mean1, mean2, n_steps, n_experiments)

print(f"Average regret at time of decision: {np.mean(regret)}, Average decision time: {np.mean(decision_time)}.")
delta = abs(mean2 - mean1)
theoretical_regret = ( (8*std**2 / delta)*c(alpha) + delta ) * np.log(1/confidence)
print(f"Theoretical regret bound: {theoretical_regret}")

"""Our implementation of the $UCB_{\alpha}$ algorithm seems to have the expected behaviour: a low $\alpha$ reduces the regret at the cost of a higher decision time, and *vice-versa*.

We do note that we capped the number of steps at 1000 in order to have reasonable computing times. This means that although tendencies are as expected, the displayed numbers for decision times have no statistical value when they get close to 1000.

## Comparison between the algorithms

Now that we have functional agents and algorithms, we are able to reproduce the experiments realized in the article for the i.i.d. case with two arms.
"""

std = 1
mean1 = 0
mean2 = 1
n_steps = 1000
n_experiments = 100
log_inv_delta_range = np.linspace(0, 10, 10)

# Simulation with UCB_alpha
alphas = [1, 2, 4, 32, 1000]

regrets_ucb = []
decision_times_ucb = []

for alpha in alphas :
    print(f"Simulating alpha = {alpha}...", end=" ")
    agent_alpha = UCBAlpha_Agent(alpha)

    regrets_alpha = []
    decision_times_alpha = []

    for log_inv_delta in log_inv_delta_range :
        delta = np.exp(-log_inv_delta)
        regret, decision_time = run_experiment(agent_alpha, delta, std, mean1, mean2, n_steps, n_experiments)

        regrets_alpha.append(regret)
        decision_times_alpha.append(decision_time)

    regrets_ucb.append(regrets_alpha)
    decision_times_ucb.append(decision_times_alpha)

    print("done !")

# Simulation with ETC
agent_etc = ETC_Agent()

regrets_etc = []
decision_times_etc = []

for log_inv_delta in log_inv_delta_range :
    delta = np.exp(-log_inv_delta)
    regret, decision_time = run_experiment(agent_etc, delta, std, mean1, mean2, n_steps, n_experiments)

    regrets_etc.append(regret)
    decision_times_etc.append(decision_time)

plt.figure(figsize=(8, 5))
plt.xlabel("log(1/δ)")
plt.ylabel("Regret")

X = log_inv_delta_range
for Y in regrets_ucb :
    plt.plot(X, Y)
plt.plot(X, regrets_etc)

ETC_regret_bound = [ (8*std**2)/(mean2-mean1) * l for l in log_inv_delta_range]
plt.plot(X, ETC_regret_bound, 'b--')

plt.legend(["alpha = 1", "alpha = 2", "alpha = 4", "alpha = 32", "alpha = 1000 (≈ ETC')", "ETC", "ETC bound"])

plt.figure(figsize=(8, 5))
plt.xlabel("log(1/δ)")
plt.ylabel("Decision time")

X = log_inv_delta_range
for Y in decision_times_ucb :
    plt.plot(X, Y)
plt.plot(X, decision_times_etc)

plt.legend(["alpha = 1", "alpha = 2", "alpha = 4", "alpha = 32", "alpha = 1000 (≈ ETC')", "ETC"])