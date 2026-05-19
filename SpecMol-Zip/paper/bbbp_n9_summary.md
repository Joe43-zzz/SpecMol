# BBBP n=9 extension summary

| Variant | n=3 mean | n=3 std | n=6 ext mean | n=6 ext std | n combined | combined mean | combined std |
|---|---|---|---|---|---|---|---|
| V2-T5 | 0.8623 | 0.0245 | 0.8150 | 0.0230 | 9 | 0.8308 | 0.0323 |
| V0 | 0.8355 | 0.0357 | 0.8240 | 0.0254 | 9 | 0.8278 | 0.0275 |
| T6 | 0.8397 | 0.0356 | 0.8260 | 0.0135 | 9 | 0.8305 | 0.0219 |

## Random-pair ablation (matched seeds 9/19/29)

- V2-T5 (real Uni-Mol pair): [0.862, 0.838, 0.887] → mean 0.8623
- V2-T5 (random Gaussian pair): ['0.8479', '0.8137', '0.8282'] → mean 0.8299
- per-seed deltas (real − random): ['+0.0141', '+0.0243', '+0.0588']
- mean delta: +0.0324
- sign test: 3/3 seeds favor real pair (binomial one-sided p≈0.125)

## n=9 BBBP grand summary

| Variant | n=9 mean | n=9 std |
|---|---|---|
| V0 | 0.8278 | 0.0275 |
| V2-T5 | 0.8308 | 0.0323 |
| T6 | 0.8305 | 0.0219 |
| V2-T5 (random pair, n=3) | 0.8299 | 0.0172 |
