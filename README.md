# MNN.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alexander-reimer.github.io/Simulation-of-MNNs/dev/)
[![Build Status](https://github.com/alexander-reimer/Simulation-of-MNNs/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexander-reimer/Simulation-of-MNNs/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alexander-reimer/Simulation-of-MNNs/branch/main/graph/badge.svg)](https://codecov.io/gh/alexander-reimer/Simulation-of-MNNs)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Static Badge](https://img.shields.io/badge/Jugend%20forscht-%F0%9F%A5%87%20nationwide-red)](https://www.jugend-forscht.de/index.php?id=262&tx_smsjufoprojects_smsjufprojectdb%5Bproject%5D=7550&tx_smsjufoprojects_smsjufprojectdb%5Baction%5D=show&tx_smsjufoprojects_smsjufprojectdb%5Bcontroller%5D=Project&cHash=0d317fddb3cee8cc358da07a5834ccaf)

We (Matteo Friedrich and Alexander Reimer) are currently participating in Jugend
forscht 2024 (a German youth science competition). For our project, we needed a
library capable of simulating, optimizing and visualizing Mechanical Neural
Networks (MNNs). Since we couldn't find one, we decided to develop and publish
our own: MNN.jl. The source code is available in our [GitHub
repository](https://github.com/Alexander-Reimer/Simulation-of-MNNs).

While this repository is hosted on the profile of Alexander Reimer
(alexander.reimer2357@gmail.com), [Matteo
Friedrich](https://github.com/MatteoFriedrich) (matteo.r.friedrich@gmail.com)
has equally contributed to this project.

## What Are MNNs?

Mechanical Neural Networks (MNNs for short) are networks made of mass points
connected by springs with variable spring constants. With a large enough
network, a MNN can be trained to exhibit almost any behaviour desired by setting
the spring constants appropiately. Importantly, they can be trained with
multiple behaviours at once to exhibit different reactions depending on the
"inputs" (forces acting on them). This makes them potentially useful for
airplane wings which adjust theirs shape depending on the force and angle of the
wind, body armor absorbing shock, better seat cushions or better wind turbine
blades. Besides these examples of shape morphing behaviours, adjusting the
resonance curves of MNNs is also possible and opens the door for applications
like earthquake safe buildings, better music instruments, objects like walls
capable of strengthening or weakening acoustic signals at will.

We based our research and knowledge of MNNs on the 2022 paper ["Mechanical
neural networks: Architected materials that learn
behaviors"](https://www.science.org/doi/10.1126/scirobotics.abq7278) by Ryan H.
Lee, Erwin A. B. Mulder and Jonathan B. Hopkins. They are, as far as we are
aware, the first to describe such Mechanical Neural Networks. Ntwork.

## Features of this library

- Simulation of MNNs
- Evaluating the performance of MNNs for shape morphing and resonance behaviour
  with MSE
- Optimizing the spring constants with Partial Pattern Search or an evolutionary
  algorithm

## Our Research

For detailed papers (only in German), see the [paper branch on
GitHub](https://github.com/Alexander-Reimer/Simulation-of-MNNs/tree/paper). The
most recent one available is from our participation on the national level:
[paper.pdf](https://github.com/Alexander-Reimer/Simulation-of-MNNs/blob/paper/2024/JuFo_bundesweit/main.pdf)
