```@meta
CurrentModule = MNN
```

# MNN

Documentation for [MNN](https://github.com/Alexander-Reimer/Simulation-of-MNNs).

This project was created for Jugend Forscht 2024 (a German STEM competition).

## Project Summary (German)

Wir wollen uns mit dem neuen, noch vergleichsweise wenig erforschten Bereich der
*mechanical neural networks*, kurz MNNs, beschäftigen. MNNs sind programmierbare
Materialien, welchen verschiedene Verhaltensweisen, wie zum Beispiel ein
bestimmtes Verformungsverhalten, antrainiert werden können. Sie bestehen aus
Massepunkten (genannt Neuronen), welche durch Federn miteinander verbunden
werden. Ihr Verhalten ergibt sich durch die Steifheiten der Federn. Die
grundlegende Annahme von MNNs ist, dass diese Federkonstanten in zukünftigen
Materialien einzeln angepasst werden können. In Analogie zu künstlichen
neuronalen Netzwerken wäre es dann prinzipiell möglich, durch eine geeignet
gewählte Konfiguration an Federkonstanten verschiedene Verhaltensweisen auf
externe Kräfte anzutrainieren. Während sich die bisherige Forschung auf die
technische, physische Implementation dieser Netzwerke fokussiert hat, wollen wir
das Trainingsverfahren optimieren. Dazu haben wir bereits eine Simulation eines
MNNs umgesetzt, die bisher verwendeten Algorithmen (evolutionäres Lernen und
Pattern Search) selbst implementiert, sowie mit neuen Parametern ausprobiert und
verglichen. Dafür haben wir uns jedoch auf die Anwendung dieser Algorithmen in
Simulationsrechnungen beschränkt. Die Ergebnisse sollten dennoch einen guten
Startpunkt für reale MNNs bieten. Der von uns entwickelte Code ist die erste
öffentlich verfügbare Implementation eines MNNs. Unsere Ergebnisse zeigen, dass
MNNs mehrere komplexe Verhaltensweisen lernen können. Diese intelligenten
Materialien eröffnen vielfältige zukünftige technologische
Anwendungsmöglichkeiten.

## Installation

Open the Julia REPL, then

```
julia> # type ] to open package manager

pkg> add https://github.com/Alexander-Reimer/Simulation-of-MNNs
```

This should install the MNN package and all dependencies automatically.

### NixOS

When using NixOS, activating the development shell is necessary to make GLMakie work:

```bash
cd /path/to/this/project
nix-shell .
```

## Usage

To use MNN.jl in you project, you should first import it with

```julia
using MNN
```

You can then create your first MNN (see [`MNN.get_user_behaviour`](@ref) for
instructions on how to use the GUI):

```julia
# create a MNN  with 5 columns and 4 rows
net = Network(5, 4) 
# show the network
vis = Visualizer(net)
# create a behaviour to simulate using a GUI
b = get_user_behaviour(network::Network)
# simulate network with forces determined by b applied for 500 time steps
simulate!(net, Diff(500), b, vis = vis)
```


```@index
```

```@autodocs
Modules = [MNN]
```
