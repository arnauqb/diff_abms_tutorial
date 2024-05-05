---
layout: home
---

# Differentiable Agent-Based Models <br>[AAMAS '24]

Welcome to the tutorial on differentiable agent-based models!

# Description

This tutorial will introduce a new paradigm for agent-based models (ABMs) that leverages automatic differentiation (AD) to efficiently compute the simulator's gradients. The tutorial will cover:

1. An overview of vanilla AD and AD methods to differentiate through discrete stochastic programs.
2. Examples of differentiable ABMs with millions of agents used to study disease spread across multiple countries.
3. State-of-the-art methods for simulating, calibrating, and analyzing differentiable ABMs.
4. Application of differentiable ABMs to epidemic modelling in New Zealand.


# Target audience

The tutorial is aimed at the broader AAMAS community, with a particular focus on researchers interested in "modeling and simulation of societies". The main focus will be on recent advances in agent-based modeling through the use of automatic differentiation and deep neural network integration.

The tutorial will be hands-on, with demonstrations in Python, and all materials will be publicly released after the conference. Some prior experience in Python and PyTorch is desirable but not required.


# Outline and Schedule

Date: Monday 6th of May 2024.

| Time | Session | Speaker |
| --- | --- | --- |
| 14:00 - 14:30 | State of ABM research and current challenges | Ayush Chopra |
| 14:30 - 16:00 | Hands-on: Build a differentiable ABM in PyTorch | Arnau Quera-Bofarull|
| 16:00 - 16:30 | Break | - |
| 16:30 - 16:45 | Gradient-assisted ABM algorithms | Ayush Chopra |
| 16:45 - 17:15 | Differentiable ABMs: a New Zealand application | Sijin Zhang |
| 17:15 - 17:30 | Hands-on: Variational Inference with differentiable ABMs | Arnau Quera-Bofarull |
| 17:30 - 17:45 | Building ABMs at scale with AgentTorch | Ayush Chopra |
| 17:45 - 18:00 | Closing Remarks | - |

# Materials

All materials for the tutorial can be found [here](https://github.com/arnauqb/diff_abms_tutorial).

Rendered notebooks:

- [1. Automatic Differentiation](01-automatic-differentiation)
- [2. Differentiating Randomness](02-differentiating-randomness)
- [3. Differentiable ABMs](03-differentiable-abm)
- [4. Variational Inference](04-variational-inference)
- [5. Predator-prey model](05-predator-prey)
- [6. AgentTorch](06-agent-torch)


# Presenters

[Arnau Quera-Bofarull](https://www.arnau.ai) is a postdoctoral researcher at the Department of Computer Science of the University of Oxford.

[Ayush Chopra](https://www.media.mit.edu/people/ayushc/overview/) is a PhD student at the MIT Media Lab.

[Sijin Zhang](https://www.esr.cri.nz/staff-profiles/sijin-zhang) is a senior data scientist at the Institute of Environmental Science and Research (New Zealand).