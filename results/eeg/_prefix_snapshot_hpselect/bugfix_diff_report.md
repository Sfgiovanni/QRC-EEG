# EEG bugfix: before vs after

Before is `results/eeg/_prefix_snapshot/`; after is the causal-preprocessing rerun. Deltas in endpoint rows are after minus before.

## 1. Endpoints

| Construction | Set | NRMSE before | NRMSE after | Delta | R2 before | R2 after | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| AB (noaux) | F | 0.4945 | 0.5446 | +0.0501 | 0.6787 | 0.6256 | -0.0531 |
| AB (noaux) | S | 0.5879 | 0.6064 | +0.0185 | 0.5648 | 0.5418 | -0.0230 |
| AB (noaux) | Z | 0.6589 | 0.6715 | +0.0126 | 0.4954 | 0.4746 | -0.0208 |
| Dual kernel | F | 0.4843 | 0.5332 | +0.0489 | 0.6852 | 0.6340 | -0.0512 |
| Dual kernel | S | 0.5412 | 0.5583 | +0.0171 | 0.6015 | 0.5800 | -0.0215 |
| Dual kernel | Z | 0.6361 | 0.6463 | +0.0102 | 0.5166 | 0.4987 | -0.0179 |
| ESN | F | 0.4824 | 0.5167 | +0.0343 | 0.6834 | 0.6354 | -0.0480 |
| ESN | S | 0.5219 | 0.5419 | +0.0200 | 0.6205 | 0.5907 | -0.0298 |
| ESN | Z | 0.6314 | 0.6523 | +0.0209 | 0.5203 | 0.4806 | -0.0397 |
| Single kernel | F | 0.4837 | 0.5334 | +0.0497 | 0.6858 | 0.6327 | -0.0531 |
| Single kernel | S | 0.5392 | 0.5578 | +0.0186 | 0.6030 | 0.5796 | -0.0234 |
| Single kernel | Z | 0.6353 | 0.6455 | +0.0102 | 0.5171 | 0.4990 | -0.0181 |
| Triangular | F | 0.4833 | 0.5324 | +0.0491 | 0.6856 | 0.6344 | -0.0512 |
| Triangular | S | 0.5416 | 0.5597 | +0.0181 | 0.5998 | 0.5780 | -0.0218 |
| Triangular | Z | 0.6353 | 0.6454 | +0.0101 | 0.5168 | 0.4992 | -0.0176 |
| Uniform | F | 0.4891 | 0.5377 | +0.0486 | 0.6796 | 0.6288 | -0.0508 |
| Uniform | S | 0.5492 | 0.5648 | +0.0156 | 0.5934 | 0.5749 | -0.0185 |
| Uniform | Z | 0.6402 | 0.6516 | +0.0114 | 0.5118 | 0.4923 | -0.0195 |

## 2. Kernel vs ESN-66 by horizon (F and Z)

Positive Delta-NRMSE means lower NRMSE for the kernel. Significance is Holm p < 0.05.

| Set | h | Delta before | CI before | Holm before | Delta after | CI after | Holm after |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 1 | -0.0062 | [-0.0098, -0.0033] | 0.0001144 | -0.0460 | [-0.0737, -0.0285] | 2.289e-05 |
| F | 2 | -0.0020 | [-0.0030, -0.0008] | 0.01352 | -0.0301 | [-0.0497, -0.0183] | 2.289e-05 |
| F | 4 | +0.0044 | [+0.0016, +0.0075] | 0.05107 | -0.0068 | [-0.0139, -0.0009] | 0.2919 |
| F | 8 | +0.0145 | [+0.0085, +0.0211] | 0.0008202 | +0.0198 | [+0.0104, +0.0297] | 0.003223 |
| Z | 1 | -0.0024 | [-0.0034, -0.0010] | 0.01289 | -0.0009 | [-0.0029, +0.0023] | 0.02522 |
| Z | 2 | -0.0008 | [-0.0024, +0.0007] | 0.7768 | +0.0045 | [-0.0003, +0.0105] | 0.461 |
| Z | 4 | +0.0038 | [+0.0012, +0.0064] | 0.0817 | +0.0224 | [+0.0088, +0.0381] | 0.04473 |
| Z | 8 | +0.0054 | [+0.0036, +0.0073] | 0.0003986 | +0.0365 | [+0.0158, +0.0607] | 0.006369 |

## 3. Kernel vs AB by horizon

Positive Delta-NRMSE means lower NRMSE for the kernel. Significance is Holm p < 0.05.

| Set | h | Delta before | CI before | Holm before | Delta after | CI after | Holm after |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 1 | +0.0146 | [+0.0116, +0.0180] | 6.866e-05 | +0.0181 | [+0.0155, +0.0205] | 6.866e-05 |
| F | 2 | +0.0178 | [+0.0132, +0.0226] | 6.866e-05 | +0.0186 | [+0.0137, +0.0232] | 0.0003738 |
| F | 4 | +0.0071 | [+0.0005, +0.0126] | 0.2363 | +0.0057 | [-0.0036, +0.0133] | 0.5849 |
| F | 8 | +0.0038 | [+0.0016, +0.0058] | 0.08371 | +0.0025 | [-0.0013, +0.0062] | 1 |
| S | 1 | +0.0637 | [+0.0575, +0.0706] | 6.866e-05 | +0.0647 | [+0.0585, +0.0709] | 6.866e-05 |
| S | 2 | +0.0889 | [+0.0790, +0.1000] | 6.866e-05 | +0.0882 | [+0.0768, +0.0994] | 6.866e-05 |
| S | 4 | +0.0455 | [+0.0295, +0.0612] | 0.001192 | +0.0465 | [+0.0243, +0.0641] | 0.0266 |
| S | 8 | -0.0033 | [-0.0088, +0.0025] | 1 | -0.0048 | [-0.0116, +0.0023] | 1 |
| Z | 1 | +0.0386 | [+0.0330, +0.0448] | 6.866e-05 | +0.0417 | [+0.0355, +0.0484] | 6.866e-05 |
| Z | 2 | +0.0437 | [+0.0375, +0.0504] | 6.866e-05 | +0.0480 | [+0.0404, +0.0557] | 6.866e-05 |
| Z | 4 | +0.0096 | [+0.0058, +0.0134] | 0.007414 | +0.0111 | [+0.0061, +0.0160] | 0.008059 |
| Z | 8 | +0.0023 | [+0.0003, +0.0046] | 1 | +0.0031 | [+0.0005, +0.0058] | 0.5849 |

## 4. Capacity

| Construction | Quadratic before | Quadratic after | Delta |
|---|---:|---:|---:|
| AB_noaux | 1.9143 | 1.9296 | +0.0154 |
| ESN | 0.0134 | 0.0087 | -0.0047 |
| dual_kernel | 2.4939 | 2.5051 | +0.0113 |
| single_kernel | 2.6139 | 2.6151 | +0.0012 |
| triangular | 2.6072 | 2.6154 | +0.0082 |
| uniform | 2.5822 | 2.5878 | +0.0056 |

Before: slope=-0.005202, nominal CI=[-0.011685, +0.001282], n reported=36 repeated rows.
After: descriptive slope=-0.006036, CI omitted, n=3 independent comparison-level gaps.

A medida quadrática iid utilizada não apresentou associação positiva detectável com os ganhos em EEG nas configurações avaliadas.

## 5. Factual verdict

- **Kernel x ESN crossover — ENFRAQUECEU.** Corrected directions match ESN at short and kernel at long horizons in 6/8 F/Z cells; Holm-significant cells in the expected direction: short=3, long=3.
- **Kernel > AB — SOBREVIVEU.** Corrected kernel-favoring cells=11/12; Holm-significant kernel-favoring cells=8/12.
- **Capacity-gain association — ENFRAQUECEU.** The nominal before analysis used n=36 repeated rows; the corrected descriptive analysis has n=3 independent gaps and no inferential CI (slope=-0.006036).
