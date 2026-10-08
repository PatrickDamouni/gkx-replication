# Memory and time of the real data path

94 signal columns, fresh process, peak RSS is cumulative. Machine: 26 GB RAM.

| step | seconds | peak RSS so far (GB) | rows |
|---|---|---|---|
| load | 1 | 3.51 | 4,482,749 |
| rank_normalize | 46 | 5.42 | 4,482,749 |
| largest train window -> float32 matrix | 2 | 5.42 | 3,342,721 |
