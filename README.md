Deep Reinforcement Learning for LiDAR-Only Robot Navigation

Project Description
This project studies how robust Deep Reinforcement Learning (DRL) models are when operating under degraded sensor conditions. It focuses on LiDAR-only navigation in simulated hallway environments and analyzes how reduced sensor quality affects performance.

While DRL models perform well in ideal simulations, real-world deployment introduces noise, lower resolution, and hardware limitations. This project simulates those challenges and evaluates how performance changes.

The goal is to identify the minimum LiDAR requirements for reliable navigation and understand how resilient DRL models are to imperfect sensor data.


Group Members
- Maria Inês Rocha (202305626)
- Martim Fernandes (202305016)
- Sebastian Sjøen-Tollaksvik (202503018)


Objectives
- Evaluate DRL performance for LiDAR-based navigation
- Simulate sensor degradation (resolution reduction and noise)
- Identify minimum sensor quality for safe navigation
- Compare a baseline model with a robustness-trained model
- Measure performance degradation under non-ideal conditions


Setup

Simulator:
- Webots

Environment:
- Simulated hallways with different layouts

Sensors and Actuators:
- LiDAR sensor
- Differential drive motors
- Sensor degradation applied via:
  - Reduced ray count
  - Gaussian noise

Software Stack:
- Python
- Gymnasium
- Stable-Baselines3


Approach

Algorithm:
A standard DRL algorithm (e.g., PPO or DQN) is used from the baseline implementation.

The algorithm is kept constant so that any performance differences are caused only by sensor degradation, not model changes.


Experimental Design

Model A (Baseline):
- Trained with perfect, high-resolution LiDAR
- No noise

Model B (Robust):
- Trained with reduced resolution and noise


Experiment 1: Resolution Reduction
- Evaluate with 100%, 50%, 25%, and 10% LiDAR resolution

Experiment 2: Noise Injection
- Evaluate with Low, Medium, and High Gaussian noise levels

Evaluation:
- 100 episodes per setup


Hypothesis

H0: There is no difference in performance between the baseline and robust model under degraded sensor conditions.


Performance Metrics

- Success Rate (%)
- Execution Time (seconds)
- Average Distance to Obstacles (meters)


Expected Outcomes

- Determine minimum LiDAR resolution for navigation
- Understand impact of sensor noise
- Evaluate whether training with degraded sensors improves robustness