# Experiment Report

## Headline
- Transformation: `black`
- Target model: `resnet50_wo_aug`
- Coverage threshold: `0.20`
- Number of disagreement inducing inputs found: `5`
- Neuron coverage achieved: `0.4964`
- Clean disagreement count: `1883`
- Clean disagreement rate: `0.1883`

## Model Performance
| Model | Saved Epoch | Val Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| resnet50_wo_aug | 20 | 0.8387 | 0.7419 | 0.8281 |
| resnet50_w_aug | 20 | 0.8279 | 0.5210 | 0.8224 |

## Coverage Change
| Model | Baseline | Generated | Added Neurons |
| --- | ---: | ---: | ---: |
| resnet50_wo_aug | 0.4890 (38126/77962) | 0.4956 (38641/77962) | +515 |
| resnet50_w_aug | 0.4936 (38485/77962) | 0.4972 (38765/77962) | +280 |

## Generated Suspicious Inputs
| Seed | True | Clean Prediction | Generated Prediction | Iters |
| --- | --- | --- | --- | ---: |
| 4238 | dog | (cat, cat) | (cat, bird) | 17 |
| 8462 | bird | (deer, deer) | (cat, deer) | 16 |
| 5433 | cat | (dog, dog) | (cat, dog) | 16 |
| 4613 | automobile | (automobile, automobile) | (automobile, truck) | 1 |
| 797 | deer | (horse, horse) | (horse, deer) | 1 |

## Saved Files
- `coverage_summary.png`
- `clean_disagreements/`
- `generated_inputs/`
