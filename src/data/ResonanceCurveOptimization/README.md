# Resonance curve optimization

How to get data:

1. Load using `using JLD2; @load "path/to/file.jld2"`
2. Now there are multiple variables of type `res`, `res1`, `res2`, ...
    - `resn[1]`: `DataFrame` (for `res` and `res2`, `amps` are not correct; they
      can be recalculated with network and MNN.calc_resonance_curve)
    - `resn[2]`: Used `Trainer` (at end of training)
    - `resn[3]`: Frequencies (matching `amps` in `DataFrame`)

- Successful attempt used in paper:
  - `ResCurveOptimization_2024-03-19_12-51-39.jld2`
    - `2024-03-19_ResCurveTrained.csv`: Calculated resonance curve with first
      network of `res` ("amplitudeUntrained") and last network of `res4`
      ("amplitudeTrained")
    - `2024-03-19_ResCurveTrained_springConstants.csv`: Spring constants of
      these two networks
    - Training time: 1350 epochs (1200 PPS (`res` + `res2`), then 150 Evolution
      (`res3` + `res4`))
