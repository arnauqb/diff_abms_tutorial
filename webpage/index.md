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
| 09:00 - 09:30 | State of ABM research and current challenges | Ayush Chopra |
| 09:30 - 10:30 | Hands-on: Build a differentiable ABM in PyTorch | Arnau Quera-Bofarull|
| 10:30 - 11:00 | Break | - |
| 11:00 - 12:00 | Gradient-assisted calibration of ABMs | Ayush & Arnau |
| 12:00 - 12:30 | Differentiable ABMs on-the-ground of the New Zealand population | Sijin Zhang |
| 12:30 - 13:00 | Closing Remarks | - |

# Materials

All materials for the tutorial can be found [here](https://github.com/arnauqb/diff_abms_tutorial).

Rendered notebooks:

- [1. Automatic Differentiation](notebooks_md/01-automatic-differentiation)
- [2. Differentiating Randomness](notebooks_md/02-differentiating-randomness)


# Presenters

[Arnau Quera-Bofarull](https://www.arnau.ai) is a postdoctoral researcher at the Department of Computer Science of the University of Oxford.

[Ayush Chopra](https://www.media.mit.edu/people/ayushc/overview/) is a PhD student at the MIT Media Lab.