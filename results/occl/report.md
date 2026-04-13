# Experiment Report

## Headline
- Transformation: `occl`
- Target model: `resnet50_wo_aug`
- Coverage threshold: `0.20`
- Number of disagreement inducing inputs found: `5`
- Neuron coverage achieved: `0.4921`
- Clean disagreement count: `1884`
- Clean disagreement rate: `0.1884`

## Model Performance
| Model | Saved Epoch | Val Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| resnet50_wo_aug | 20 | 0.8387 | 0.7419 | 0.8280 |
| resnet50_w_aug | 20 | 0.8279 | 0.5210 | 0.8223 |

## Coverage Change
| Model | Baseline | Generated | Added Neurons |
| --- | ---: | ---: | ---: |
| resnet50_wo_aug | 0.4890 (38123/77962) | 0.4897 (38180/77962) | +57 |
| resnet50_w_aug | 0.4936 (38485/77962) | 0.4944 (38543/77962) | +58 |

## Generated Suspicious Inputs
| Seed | True | Clean Prediction | Generated Prediction | Iters |
| --- | --- | --- | --- | ---: |
| 7153 | horse | (horse, horse) | (airplane, horse) | 19 |
| 6006 | airplane | (airplane, airplane) | (horse, airplane) | 12 |
| 341 | frog | (frog, frog) | (cat, frog) | 5 |
| 3773 | bird | (bird, bird) | (dog, bird) | 2 |
| 4441 | automobile | (truck, truck) | (automobile, truck) | 3 |

## Saved Files
- `coverage_summary.png`
- `clean_disagreements/`
- `generated_inputs/`
