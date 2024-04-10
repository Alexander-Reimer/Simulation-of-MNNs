```@meta
CurrentModule = MNN
```

# MNN.jl

!!! warning
    Please note that this project is still in development. It is already usable,
    but there will probably be frequent breaking changes in the future as we
    expand and refactor this library.

We (Matteo Friedrich and Alexander Reimer) are currently participating in Jugend
forscht 2024 (a German youth science competition). For our project, we needed a
library capable of simulating, optimizing and visualizing Mechanical Neural
Networks (MNNs). Since we couldn't find one, we decided to develop and publish
our own: MNN.jl. The source code is available in our [GitHub
repository](https://github.com/Alexander-Reimer/Simulation-of-MNNs).

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
aware, the first to describe such Mechanical Neural Networks.

## Features of this library

- Simulation of MNNs
- Evaluating the performance of MNNs for shape morphing and resonance behaviour
  with MSE
- Optimizing the spring constants with Partial Pattern Search or an evolutionary
  algorithm

## Our Research

For detailed papers (only in German), see the [paper branch on
GitHub](https://github.com/Alexander-Reimer/Simulation-of-MNNs/tree/paper). The
most recent one available is from our participation at the state-level
competition in Lower Saxony:
[paper.pdf](https://github.com/Alexander-Reimer/Simulation-of-MNNs/blob/paper/2024/JuFo_Niedersachsen/main.pdf)

### Summary

#### What We Did

- Both confirmed existing results and how representative our simulation is by
  analysing the correlation between different hyperparameters and training
  success, which mostly fit the existing research by Lee et al.
  - Unlike Lee et al., we used the optimization algorithms applicable to
    physical MNNs like Partial Pattern Search and/or evolutionary algorithms
    instead of a gradient descent based approach
- Successfully optimized the resonance curve of MNNs; we believe to be the first
  to do this, as we couldn't find anything else available online

#### What We Didn't Do (Yet)

- Build our own MNN mechanically
- Build a ressource intensive, physically accurate simulation
  - the goal is to provide starting points for the optimization of real MNNs and
    figure out the effects of parameters & different optimization algorithms
  - specifically, we haven't (yet) incorparated checking spring and neuron
    position limits, meaning that using very large input forces / goal position
    vectors can result in mass points and springs phasing through each other

## Project Summary (German)

Wir haben uns mit dem neuen Bereich der *mechanical neural networks* (MNNs)
beschäftigt - programmierbare Materialien, die aus mit Federn verbundenen
Massenpunkten bestehen. Ihnen können durch Anpassung der Federhärten
verschiedene Verhaltensweisen gleichzeitig antrainiert werden, was viele
Anwendungsmöglichkeiten eröffnet, wie z.B. Flugzeugflügel, deren Form sich
optimal an Windgeschwindigkeit und -richtung anpasst. In unserem Projekt haben
wir die Trainingsverfahren und den Einfluss verschiedener Parameter analysiert
und dafür in einer eigenen Softwarebibliothek die Simulation, Optimierung,
Bewertung und Visualisierung von MNNs umgesetzt. Außerdem haben wir als Erste
erfolgreich die Resonanzkurven von MNNs optimiert, was auch den Einsatz für z.B.
erdbebensichere Strukturen ermöglicht. Die von uns erkannten Einflüsse von
Parametern decken sich, wo vorhanden, mit der bisherigen Forschung.
