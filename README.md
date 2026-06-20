# Deep Reinforcement Learning for Robust LiDAR-Based Autonomous Navigation Research

### Investigating Sim-to-Real Robustness, Sensor Degradation, and Spatial Generalization in Autonomous Mobile Robots

This project explores one of the most important unsolved problems in robotics:

> Why do reinforcement learning agents that perform perfectly in simulation often fail in the real world?

To answer this question, I designed and conducted a multi-phase experimental study evaluating Deep Reinforcement Learning (DRL) navigation policies under increasingly realistic operating conditions.

The project benchmarks **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)** across:

* Sensor noise degradation
* LiDAR resolution reduction
* Novel environment transfer
* Spatial generalization challenges
* Dynamic obstacle scenarios

The resulting experiments reveal a surprising tradeoff:

* Policies can be highly robust to severe sensor degradation.
* The same policies can completely fail when exposed to previously unseen environments.

This highlights a key challenge in autonomous robotics:

**Generalization is often a harder problem than perception.**

---

# Highlights

### Research Outcomes

* Trained navigation policies for **2,000,000+ simulation timesteps**
* Evaluated **PPO and DQN** under multiple environment distributions
* Achieved **100% navigation success** with LiDAR resolution reduced by **90%**
* Identified sensor noise thresholds causing complete policy collapse
* Demonstrated severe zero-shot transfer failure despite perfect training performance
* Achieved **65% success rate** in complex environments using DQN
* Built a complete robotics experimentation pipeline using Webots, Stable-Baselines3, and ZeroMQ

---

# Motivation

Modern autonomous systems rely heavily on simulation because collecting real-world robotic training data is expensive, slow, and potentially dangerous.

However, simulated environments are fundamentally cleaner than reality.

Real sensors suffer from:

* Measurement noise
* Resolution limitations
* Hardware inconsistencies
* Environmental uncertainty

As a result, policies trained in simulation frequently overfit to ideal conditions.

This project investigates how navigation policies behave when those assumptions are systematically broken.

---

# Research Questions

The project was structured around three central questions:

### 1. How much sensor noise can a policy tolerate?

Can Gaussian noise improve robustness?

At what point does noise stop acting as regularization and begin destroying the learning signal?

---

### 2. Is high-resolution LiDAR actually necessary?

Can low-cost sensors achieve comparable performance?

How much information is truly required for safe navigation?

---

### 3. Can trained policies generalize to new environments?

Do navigation strategies learned in one hallway transfer to:

* bottlenecks
* sharp turns
* obstacle-rich layouts
* unseen topologies

without retraining?

---

# System Architecture

```text
Webots Simulation Environment
            │
            ▼
      LiDAR Observations
            │
            ▼
      Feature Extraction
        CNN Encoder
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
    PPO           DQN
Continuous    Discrete Actions
 Control
     │             │
     └──────┬──────┘
            ▼
      Velocity Commands
            │
            ▼
 Differential Drive Robot
```

---

# Experimental Design

## Phase 1 — Sensor Noise Robustness

The first phase focused on understanding how noise influences policy learning.

Two training regimes were evaluated:

| Model           | Configuration      |
| --------------- | ------------------ |
| Baseline PPO    | 360 Rays, No Noise |
| Stress-Test PPO | 360 Rays, σ = 0.5  |

The objective was to identify the point at which noise transitions from useful regularization into state-space corruption.

### Key Result

Training collapsed entirely at high noise levels.

Noise beyond approximately σ = 0.2 dramatically degraded performance and prevented stable policy formation.

---

## Phase 2 — LiDAR Resolution Study

The second phase investigated whether high-density LiDAR scans are actually required.

Two independent agents were trained:

| Model | Training Resolution |
| ----- | ------------------- |
| PPO-C | 360 Rays            |
| PPO-D | 90 Rays             |

Agents were then evaluated under progressively reduced sensor configurations.

### Key Result

Both agents achieved:

**100% success rate across every tested resolution**

including configurations as low as:

* 36 rays
* 90 rays
* 180 rays
* 360 rays

This suggests navigation policies primarily learn geometric structure rather than relying on dense angular sampling.

The finding has practical implications for:

* Autonomous wheelchairs
* Service robots
* Low-cost mobile platforms
* Embedded robotic systems

where computational efficiency and sensor cost are critical constraints.

---

## Phase 3 — Spatial Generalization

The most challenging phase evaluated transfer to unseen environments.

The original training map was expanded with:

* Tight bottlenecks
* Static pillars
* Sharp cornering scenarios
* Constricted corridors
* U-turn sections
* Novel agent spawn locations

No prior exposure to these layouts was provided during initial training.

### Zero-Shot Transfer Results

| Algorithm | Success Rate |
| --------- | ------------ |
| PPO       | 8.24%        |

Despite near-perfect performance in the original environment, the agent failed catastrophically when confronted with unfamiliar spatial structures.

This exposed severe environment-specific overfitting.

---

### PPO Fine-Tuning

Additional training improved performance:

| Algorithm        | Success Rate |
| ---------------- | ------------ |
| PPO (Fine-Tuned) | 46%          |

However, adaptation remained inconsistent across navigation tasks.

---

### DQN Benchmark

A DQN agent trained directly on the complex environment achieved:

| Algorithm | Success Rate |
| --------- | ------------ |
| DQN       | 65%          |

Interestingly, DQN required less total training while outperforming PPO.

This suggests discrete decision-making may offer advantages in highly constrained navigation problems.

---

# Dynamic Obstacle Evaluation

To evaluate temporal awareness, a moving obstacle was introduced into the environment.

The PPO policy consistently failed to react to the moving object.

Instead, it continued following its learned trajectory until collision occurred.

This experiment revealed an important limitation:

The policy successfully understands **space** but not **motion**.

Without temporal context, dynamic obstacles become fundamentally difficult to predict.

---

# Technology Stack

### Robotics

* Webots

### Reinforcement Learning

* Stable-Baselines3
* PPO
* DQN

### Machine Learning

* CNN Feature Extractors
* Policy Gradient Methods
* Value-Based Reinforcement Learning

### Communication

* ZeroMQ

### Programming

* Python

---

# Lessons Learned

This project challenged several assumptions commonly made in robotics.

### What mattered less than expected

* LiDAR resolution
* Sensor density
* Sparse perception

### What mattered more than expected

* Environment diversity
* Spatial generalization
* Transfer robustness
* Reward shaping

The results suggest that building navigation systems capable of adapting to unseen environments remains a significantly harder problem than handling noisy sensors.

---

# Future Work

Potential extensions include:

* Recurrent PPO (LSTM / GRU)
* Transformer-based perception models
* Domain randomization pipelines
* Curriculum learning
* Multi-sensor fusion
* Dynamic obstacle prediction
* Sim-to-real deployment
* Real robotic hardware validation

---

# Selected Results

| Experiment                     | Result        |
| ------------------------------ | ------------- |
| Maximum Training Budget        | 2M+ Timesteps |
| Lowest Tested LiDAR Resolution | 36 Rays       |
| Resolution Robustness          | 100% Success  |
| PPO Zero-Shot Transfer         | 8.24%         |
| PPO Fine-Tuned Transfer        | 46%           |
| DQN Complex Environment        | 65%           |
| Dynamic Obstacle Awareness     | Failed        |

---

# References

This project builds upon research and tooling from:

* Stable-Baselines3
* OpenAI Gym
* PPO (Schulman et al.)
* DQN (Mnih et al.)
* Webots Robotics Simulator

---

## Repository Structure

```text
.
├── controllers/
├── environments/
├── models/
├── training/
├── evaluation/
├── results/
├── plots/
├── assets/
└── README.md
```
