> **EXPLORATORY (D049).** The 1987–2016 test window was seen at `result-v1`. Nothing below is a confirmatory test or changes the D048 verdict.

# Phase 5 power (input to D050)

Code `90bf867`. 4000 block-bootstrap draws per cell (block 12 months, seed 20261012), one-sided HAC t with 5 lags, power = share p ≤ 0.05. effect×1 assumes the 1987–2016 mean differential persists unchanged (optimistic: winner's curse); effect×0 is the false-positive rate under no effect.

| test                                                 |   1987–2016 t (T=360) |   T=48, effect×1 |   T=48, effect×0.5 |   T=48, effect×0 |   T=60, effect×1 |   T=60, effect×0.5 |   T=60, effect×0 |   T=96, effect×1 |   T=96, effect×0.5 |   T=96, effect×0 |
|:-----------------------------------------------------|----------------------:|-----------------:|-------------------:|-----------------:|-----------------:|-------------------:|-----------------:|-----------------:|-------------------:|-----------------:|
| test 1: NN3 vs constant (raw DM, frozen family)      |                 2.213 |            0.204 |              0.096 |            0.04  |            0.23  |              0.099 |            0.036 |            0.298 |              0.112 |            0.031 |
| NN3 vs ENet+H (raw DM, frozen family)                |                 1.319 |            0.116 |              0.057 |            0.028 |            0.123 |              0.059 |            0.026 |            0.144 |              0.066 |            0.029 |
| NN3-dm vs constant (raw DM)                          |                 4.568 |            0.638 |              0.306 |            0.073 |            0.694 |              0.32  |            0.062 |            0.828 |              0.374 |            0.051 |
| NN3-dm vs ENet+H (within-month DM, frozen secondary) |                 4.034 |            0.543 |              0.204 |            0.041 |            0.604 |              0.222 |            0.039 |            0.784 |              0.278 |            0.034 |
| NN3 vs ENet+H (within-month DM)                      |                 2.42  |            0.264 |              0.108 |            0.034 |            0.288 |              0.108 |            0.032 |            0.374 |              0.131 |            0.031 |
