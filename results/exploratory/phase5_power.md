> **EXPLORATORY (D049).** The 1987–2016 test window was seen at `result-v1`. Nothing below is a confirmatory test or changes the D048 verdict.

# Phase 5 power (input to D050)

Code `ad88b07`. 4000 block-bootstrap draws per cell (block 12 months, seed 20261012), one-sided HAC t with 5 lags, power = share p ≤ 0.05. effect×1 assumes the 1987–2016 mean differential persists unchanged (optimistic: winner's curse); effect×0 is the false-positive rate under no effect.

| test                                                 |   1987–2016 t (T=360) |   T=48, effect×1 |   T=48, effect×0.5 |   T=48, effect×0 |   T=60, effect×1 |   T=60, effect×0.5 |   T=60, effect×0 |   T=96, effect×1 |   T=96, effect×0.5 |   T=96, effect×0 |
|:-----------------------------------------------------|----------------------:|-----------------:|-------------------:|-----------------:|-----------------:|-------------------:|-----------------:|-----------------:|-------------------:|-----------------:|
| test 1: NN3 vs constant (raw DM, frozen family)      |                 2.213 |            0.202 |              0.092 |            0.039 |            0.23  |              0.098 |            0.037 |            0.298 |              0.111 |            0.032 |
| NN3 vs ENet+H (raw DM, frozen family)                |                 1.319 |            0.116 |              0.056 |            0.027 |            0.121 |              0.06  |            0.026 |            0.144 |              0.066 |            0.029 |
| NN3-dm vs constant (raw DM)                          |                 4.569 |            0.638 |              0.302 |            0.072 |            0.696 |              0.318 |            0.06  |            0.829 |              0.376 |            0.05  |
| NN3-dm vs ENet+H (within-month DM, frozen secondary) |                 4.037 |            0.543 |              0.204 |            0.041 |            0.606 |              0.222 |            0.039 |            0.786 |              0.28  |            0.034 |
| NN3 vs ENet+H (within-month DM)                      |                 2.422 |            0.266 |              0.109 |            0.033 |            0.289 |              0.108 |            0.032 |            0.375 |              0.133 |            0.031 |
