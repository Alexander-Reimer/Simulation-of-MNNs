# MNN

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alexander-reimer.github.io/Simulation-of-MNNs/dev/)
[![Build Status](https://github.com/alexander-reimer/Simulation-of-MNNs/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alexander-reimer/Simulation-of-MNNs/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alexander-reimer/Simulation-of-MNNs/branch/main/graph/badge.svg)](https://codecov.io/gh/alexander-reimer/Simulation-of-MNNs)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

The German papers describing this project can be found in the [paper
branch](https://github.com/Alexander-Reimer/Simulation-of-MNNs/tree/paper).

## Summary (German)

Wir wollen uns mit dem neuen, vergleichsweise wenig erforschtem Bereich der
Mechanical Neural Networks, kurz MNNs, beschäftigen. Während die bisherige
Forschung sich auf die technische, physische Implementation dieser Netzwerke
fokussiert hat, wollen wir das Trainingsverfahren optimieren. Dazu wollen wir
die bisher verwendeten Algorithmen (evolutionäres Lernen und Pattern Search)
selbst implementieren und genauer sowie mit neuen Parametern auszuprobieren und
zu vergleichen. Außerdem wollen wir versuchen, eine Methode für Backpropagation
bei MNNs zu entwickeln und implementieren, da eine solche unserer Recherche nach
noch nicht ausprobiert wurde.

Dafür werden wir uns jedoch auf die Anwendung dieser Algorithmen in einer
Simulation beschränken. Die Ergebnisse sollten dennoch einen guten Startpunkt
für reale MNNs bieten.

## Usage

### NixOS

When using NixOS, activating the development shell is necessary to make GLMakie work:

```bash
cd /path/to/this/project
nix-shell .
```
