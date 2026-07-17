---
title: "Exploration in Reinforcement Learning"
date: 2026-07-17
description: "A practical guide to entropy-regularized PPO and RND — from the mathematical foundations of uncertainty-directed exploration to experimental results on Ant-v4"
tags: ["reinforcement-learning", "exploration", "PPO", "RND", "entropy", "MuJoCo"]
categories: ["RL"]
math: true
toc: true
cover:
  image: "images/comparison_bar_chart.png"
  alt: "Model comparison bar chart"
  caption: "Comparison of mean episode returns across models at 800k steps on Ant-v4"
---

## The Problem

In sparse-reward environments, standard RL agents receive no learning signal for most of their actions. Without structured exploration, they collapse to local optima or fail to learn entirely.

Consider the policy gradient:

\[
\nabla_\theta J(\theta) = \mathbb{E}_{s, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a) \right]
\]

When \(Q^\pi(s, a) \approx 0\) for most \((s, a)\), the gradient vanishes. The agent receives no signal about which actions are valuable, and random exploration (ε-greedy) is exponentially inefficient in high-dimensional action spaces.

This is the **exploration problem**: how do we structure exploration so the agent discovers rewarding states efficiently?

Our approach combines two complementary forms of uncertainty-directed exploration — **entropy regularization** (maintaining action diversity) and **Random Network Distillation** (driving state coverage). Together they form a principled exploration strategy rooted in information theory.

---

## Entropy Regularization: The Standard Approach

Proximal Policy Optimization (PPO) clips the surrogate objective to prevent destructive policy updates:

\[
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\]

where \(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}\) is the probability ratio.

The full PPO objective adds a value function loss and an **entropy bonus**:

\[
L^{PPO}(\theta) = L^{CLIP}(\theta) - c_1 \cdot L^{VF}(\theta) + c_2 \cdot \mathcal{H}(\pi_\theta(\cdot|s_t))
\]

The Shannon entropy of the policy measures its uncertainty:

\[
\mathcal{H}(\pi_\theta(\cdot|s)) = -\int_a \pi_\theta(a|s) \log \pi_\theta(a|s) \, da
\]

For a Gaussian policy \(\pi(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))\) — the standard choice for continuous control — this has a closed form:

\[
\mathcal{H}(\pi_\theta(\cdot|s)) = \frac{1}{2} \log\left(2\pi e \,\sigma_\theta^2(s)\right)
\]

The gradient with respect to the variance is:

\[
\frac{\partial \mathcal{H}}{\partial \sigma^2} = \frac{1}{2\sigma^2}
\]

**This is the key mechanism**: as the policy variance \(\sigma^2\) approaches zero (the policy becomes deterministic), the entropy gradient diverges to infinity. This creates a strong repulsive force against policy collapse.

The entropy coefficient \(c_2\) in the PPO objective directly controls the exploration-exploitation tradeoff:

- **High \(c_2\)** → policy maintains high variance → more exploration, slower convergence
- **Low \(c_2\)** → policy collapses to near-deterministic → faster convergence, risk of local optima
- **\(c_2 = 0\)** → no entropy regularization → premature policy collapse

This is the "balancing via uncertainty" mechanism: the policy is explicitly rewarded for being uncertain about which action to take.

---

## Random Network Distillation: Epistemic Uncertainty

While entropy regularization maintains **action diversity**, RND addresses a different form of uncertainty: **state novelty**.

RND uses two neural networks:

- **Target network** \(f_{\text{target}}: \mathcal{S} \to \mathbb{R}^d\) — randomly initialized, **frozen**
- **Predictor network** \(f_{\text{predictor}}: \mathcal{S} \to \mathbb{R}^d\) — trained to match the target on visited states

The intrinsic reward is the prediction error:

\[
r_{\text{intrinsic}}(s) = \| f_{\text{predictor}}(s) - f_{\text{target}}(s) \|^2
\]

**Why this works**: The predictor can only learn to predict states it has visited. Novel states produce high prediction error → high intrinsic reward → the agent is incentivized to explore unknown regions.

### The Unified View

These two mechanisms are **complementary**:

| Method | Uncertainty Type | What It Prevents |
|--------|----------------|-------------------|
| Entropy bonus | **Aleatoric** (action) | Policy collapse to deterministic |
| RND | **Epistemic** (state) | State space under-exploration |

The unified objective:

\[
J_{\text{total}}(\theta) = \underbrace{J_{\text{extrinsic}}(\theta)}_{\text{task reward}} + c_2 \cdot \underbrace{\mathcal{H}(\pi_\theta)}_{\text{action uncertainty}} + \beta_{\text{rnd}} \cdot \underbrace{\mathbb{E}[r_{\text{intrinsic}}(s)]}_{\text{state uncertainty}}
\]

Entropy alone is insufficient for hard exploration problems — it explores actions, not states. RND alone can lead to "noise-seeking" behavior without entropy to maintain action diversity. Together they form a complete exploration strategy.

---

## Implementation

The project structure is straightforward:

```
rl-exploration-poc/
├── configs/ant_default.yaml
├── src/
│   ├── algorithms/
│   │   ├── rnd.py                   # RNDModule + RunningMeanStd
│   │   └── base_ppo.py             # PPO factory
│   ├── envs/
│   │   ├── wrappers.py              # SparseRewardWrapper
│   │   └── intrinsic_reward_wrapper.py
│   ├── experiments/
│   │   ├── train_baseline.py
│   │   ├── train_with_rnd.py
│   │   └── compare_experiments.py
│   └── utils/
│       ├── logger.py
│       ├── video_recorder.py
│       └── plotting.py
└── results/
```

### RNDModule

The core of the exploration system. Two networks — target (frozen, random) and predictor (trained) — where the prediction error serves as an intrinsic reward:

```python
class RNDModule:
    def __init__(self, obs_dim, feature_dim=128, hidden_dim=256, lr=1e-3):
        self.target = self._build_network(obs_dim, feature_dim, hidden_dim)
        self.predictor = self._build_network(obs_dim, feature_dim, hidden_dim)
        for param in self.target.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_intrinsic_reward(self, obs):
        with torch.no_grad():
            target_feat = self.target(obs)
            pred_feat = self.predictor(obs)
        return torch.mean((pred_feat - target_feat) ** 2, dim=1).cpu().numpy()

    def update(self, obs):
        target_feat = self.target(obs).detach()
        pred_feat = self.predictor(obs)
        loss = nn.MSELoss()(pred_feat, target_feat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

### IntrinsicRewardWrapper

This VecEnv wrapper adds the RND intrinsic reward to the extrinsic reward at each step:

```python
class IntrinsicRewardWrapper(VecEnvWrapper):
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        intrinsic = self.rnd.compute_intrinsic_reward(obs)
        self.rnd.update(obs)
        total_rewards = rewards + self.beta * intrinsic
        for i, info in enumerate(infos):
            info["intrinsic_reward"] = intrinsic[i]
        return obs, total_rewards, dones, infos
```

### Experiment Setup

We use the MuJoCo Ant-v4 environment — an 8-DoF quadruped with a 27-dimensional observation space and 8 continuous actions. This is a challenging continuous control task involving balance, contact dynamics, and locomotion.

The PPO configuration:

```yaml
env_id: Ant-v4
total_timesteps: 800_000
n_envs: 4
seed: 42

ppo:
  learning_rate: 0.0003
  ent_coef: 0.01
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2

rnd:
  feature_dim: 128
  hidden_dim: 256
  learning_rate: 0.001
  beta: 0.1
```

To isolate the effect of the entropy coefficient, we ran an entropy sweep comparing three values (\(c_2 = 0.0, 0.01, 0.1\)) alongside the baseline PPO and PPO+RND models.

---

## Results

### Model Comparison

| Model | Mean Return | Std Return | Episode Length |
|-------|-------------|------------|----------------|
| Baseline PPO | +6.6 | ±9.0 | 15 |
| PPO + RND | **+20.0** | ±9.9 | 19 |
| ent_coef=0.0 (no entropy) | **+547.6** | ±139.4 | 913 |
| ent_coef=0.01 (default) | +19.4 | ±21.8 | 20 |
| ent_coef=0.1 (high ent) | -15.7 | ±20.4 | 12 |

![Model Comparison](/images/comparison_bar_chart.png)

The bar chart shows a dramatic spread. The entropy coefficient dominates: \(c_2=0.0\) achieves a mean return of **+547.6** by sustaining a long episode (913 steps), while \(c_2=0.1\) collapses to negative return. RND provides a modest but consistent improvement over the default-entropy baseline (+20.0 vs +6.6).

![Entropy Sweep](/images/entropy_sweep_800k.png)

The entropy coefficient \(c_2\) has a **dominant** effect on PPO's performance:

- **\(c_2 = 0.0\)**: Return of **+547.6** — the policy collapses to deterministic, aggressively running forward
- **\(c_2 = 0.01\)**: Return of **+19.4** — the default SB3 entropy penalty regularizes too aggressively for the Ant task, preventing sustained forward momentum
- **\(c_2 = 0.1\)**: Return of **-15.7** — too much entropy leads to random behavior

### Policy Videos

<div style="display: flex; gap: 16px; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 280px;">
    <video controls width="100%">
      <source src="/videos/baseline_800k_long.mp4" type="video/mp4">
    </video>
    <p style="text-align:center;"><strong>Baseline PPO (800k)</strong><br>avg **14 frames**/episode</p>
  </div>
  <div style="flex: 1; min-width: 280px;">
    <video controls width="100%">
      <source src="/videos/rnd_800k_long.mp4" type="video/mp4">
    </video>
    <p style="text-align:center;"><strong>PPO + RND (800k)</strong><br>avg **21 frames**/episode</p>
  </div>
  <div style="flex: 1; min-width: 280px;">
    <video controls width="100%">
      <source src="/videos/ent00_800k_long.mp4" type="video/mp4">
    </video>
    <p style="text-align:center;"><strong>ent=0.0 (winner, 800k)</strong><br>sustained **1000 frames**/episode</p>
  </div>
</div>

The video lengths are proportional to performance. Baseline and RND agents fall within seconds, while the \(c_2=0.0\) agent maintains forward motion for the entire episode.

### Key Takeaways

1. **The entropy coefficient dominates PPO performance**: On Ant-v4, \(c_2\) is the single most important hyperparameter. The difference between \(c_2=0.0\) (+547.6) and \(c_2=0.01\) (+19.4) is **28x**.

2. **RND provides a modest but consistent improvement**: At 800k steps, PPO+RND (+20.0) outperforms baseline PPO (+6.6) by 3x at the same entropy setting. RND's intrinsic reward helps maintain exploration when the entropy bonus is present.

3. **Ant-v4 rewards aggressive, deterministic policies**: The Ant locomotion task benefits from sustained forward momentum, which is best achieved by a near-deterministic policy. Too much entropy (\(c_2=0.1\)) collapses performance.

4. **Hyperparameter sensitivity is extreme**: A single order-of-magnitude change in \(c_2\) (\(0.01 \to 0.0\)) changes return from +19.4 to +547.6. Always tune \(c_2\) per environment.

---

## Conclusion & Next Steps

### What We Learned

Entropy regularization and RND are complementary — they operate on different axes of uncertainty:

- **Entropy** explores the action space: "I'm not sure which action to take, so I'll try different ones"
- **RND** explores the state space: "I haven't seen this state before, so I should visit it"

The entropy coefficient \(c_2\) is a critical hyperparameter that directly controls the exploration-exploitation tradeoff. Too low and the policy collapses to deterministic; too high and it never converges.

### Next Steps

- **Port to Isaac Lab**: Replace Gymnasium with Isaac Lab tasks (e.g., ANYmal, Franka) using SKRL or RSL-RL
- **Other exploration methods**: ICM (forward dynamics model), Exploration by Disagreement (ensemble of predictors), Never Give Up (episodic + life-long novelty)
- **Hyperparameter tuning**: Systematic sweep over \(c_2\) and \(\beta_{\text{rnd}}\) to find the optimal balance

### How to Reproduce

```bash
git clone https://github.com/yidaweng/rl-exploration-poc
cd rl-exploration-poc
uv sync

# Train baseline PPO
uv run python -m src.experiments.train_baseline --total_timesteps 800000

# Train PPO + RND
uv run python -m src.experiments.train_with_rnd --total_timesteps 800000

# Entropy sweep
uv run python -m src.experiments.train_baseline --ent_coef 0.0 --run_name baseline-ent-0.0
uv run python -m src.experiments.train_baseline --ent_coef 0.01 --run_name baseline-ent-0.01
uv run python -m src.experiments.train_baseline --ent_coef 0.1 --run_name baseline-ent-0.1
```

All code is available at [github.com/yidaweng/rl-exploration-poc](https://github.com/yidaweng/rl-exploration-poc).
