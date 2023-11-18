# MNN

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://alex.github.io/MNN.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://alex.github.io/MNN.jl/dev/)
[![Build Status](https://github.com/alex/MNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alex/MNN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/alex/MNN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/alex/MNN.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

TODO-Liste: <https://github.com/users/Alexander-Reimer/projects/11>

Bei Verwendung von NixOS notwendig vorher in Shell, damit Makie funktioniert:

```bash
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
```

## Zusammenfassung

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

## Benchmarking

No changes:

- Median 78ms
- 74ms ... 94ms
- Mem. est.: 117MiB,
- Allocs est.: 1.5 million

Replace collect with loop:

- Median 74ms
- 59ms ... 96ms
- Mem. est.: 108MiB,
- Allocs est.: 1.4 million


Change 3:

- Replaced
    ```julia
    accelerations[:,v] = f
    ```
    with
    ```julia
    accelerations[:,v][1] = f[1]
    accelerations[:,v][2] = f[2]
    ```
- Median 46 μs
- 42.116 μs ... 7.318 ms
- Mem. est.: 92 KiB
- Allocs est.: 1196
