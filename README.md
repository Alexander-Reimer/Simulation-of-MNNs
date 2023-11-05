# MNN

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://alex.github.io/MNN.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex.github.io/MNN.jl/dev/)
[![Build Status](https://github.com/alex/MNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alex/MNN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alex/MNN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex/MNN.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Notwendig vorher in Shell:

```bash
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
```

## Zusammenfassung

Wir wollen uns mit dem neuen, vergleichsweise wenig erforschtem Bereich der Mechanical Neural Networks, kurz MNNs, beschäftigen. Während die bisherige Forschung sich auf die technische, physische Implementation dieser Netzwerke fokussiert hat, wollen wir das Trainingsverfahren optimieren. Dazu wollen wir die bisher verwendeten Algorithmen (evolutionäres Lernen und Pattern Search) selbst implementieren und genauer sowie mit neuen Parametern auszuprobieren und zu vergleichen. Außerdem wollen wir versuchen, eine Methode für Backpropagation bei MNNs zu entwickeln und implementieren, da eine solche unserer Recherche nach noch nicht ausprobiert wurde.

Dafür werden wir uns jedoch auf die Anwendung dieser Algorithmen in einer Simulation beschränken; die Ergebnisse sollten dennoch einen guten Startpunkt für reale MNNs bieten.