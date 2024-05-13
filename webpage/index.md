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

The tutorial will feature demonstrations in Python, and all materials will be publicly released after the conference. Some prior experience in Python and PyTorch is desirable but not required.


# Outline and Schedule

Date: Monday 6th of May 2024.

| Time | Session | Speaker |
| --- | --- | --- |
| 14:00 - 14:30 | State of ABM research and current challenges | Ayush Chopra |
| 14:30 - 15:30 | Demonstration: Build a differentiable ABM in PyTorch | Arnau Quera-Bofarull|
| 15:30 - 16:00 | Gradient-assisted ABM algorithms | Ayush Chopra |
| 16:00 - 16:30 | Break | - |
| 16:30 - 17:00 | Differentiable ABMs: a New Zealand application | Sijin Zhang |
| 17:00 - 17:25 | Demonstration: Variational Inference with differentiable ABMs | Arnau Quera-Bofarull |
| 17:25 - 17:50 | Demonstration: Building ABMs at scale with AgentTorch | Ayush Chopra |
| 17:50 - 18:00 | Closing Remarks | - |

# Materials

You can access the slides [here](webpage/AAMAS_Tutorial.pdf)

The Jupyter notebooks for the tutorial can be found in the `tutorials` directory. You can access the google colab and markdown-rendered versions here:

1. Automatic Differentiation: [[Colab]](https://colab.research.google.com/github/arnauqb/diff_abms_tutorial/blob/main/notebooks/01-automatic-differentiation.ipynb) [[Markdown]](01-automatic-differentiation)

2. Differentiating Randomness: [[Colab]](https://colab.research.google.com/github/arnauqb/diff_abms_tutorial/blob/main/notebooks/02-differentiating-randomness.ipynb)[[Markdown]](02-differentiating-randomness)
3. Differentiable ABMs: [[Colab]](https://colab.research.google.com/github/arnauqb/diff_abms_tutorial/blob/main/notebooks/03-differentiable-abm.ipynb)[[Markdown]](03-differentiable-abm)
4. Variational Inference: [[Colab]](https://colab.research.google.com/github/arnauqb/diff_abms_tutorial/blob/main/notebooks/04-variational-inference.ipynb)[[Markdown]](04-variational-inference)
5. Introduction to AgentTorch: [[Colab]](https://colab.research.google.com/github/arnauqb/diff_abms_tutorial/blob/main/notebooks/05-predator-prey.ipynb)[[Markdown]](05-predator-prey)
6. Advanced AgentTorch API: [[Colab]](https://colab.research.google.com/github/arnauqb/diff_abms_tutorial/blob/main/notebooks/06-agent-torch.ipynb)[[Markdown]](06-agent-torch)



# Presenters

[Arnau Quera-Bofarull](https://www.arnau.ai) is a postdoctoral researcher at the Department of Computer Science of the University of Oxford.

[Ayush Chopra](https://www.media.mit.edu/people/ayushc/overview/) is a PhD student at the MIT Media Lab.

[Sijin Zhang](https://www.esr.cri.nz/staff-profiles/sijin-zhang) is a senior data scientist at the Institute of Environmental Science and Research (New Zealand).