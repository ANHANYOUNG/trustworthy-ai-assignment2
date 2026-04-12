# Experiment Report

## Headline
- Transformation: `occl`
- Target model: `resnet50_wo_aug`
- Coverage threshold: `0.20`
- Number of disagreement inducing inputs found: `5`
- Neuron coverage achieved: `0.4919`
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
| resnet50_wo_aug | 0.4890 (38126/77962) | 0.4894 (38157/77962) | +31 |
| resnet50_w_aug | 0.4936 (38485/77962) | 0.4943 (38535/77962) | +50 |

## Generated Suspicious Inputs
| Seed | True | Clean Prediction | Generated Prediction | Iters |
| --- | --- | --- | --- | ---: |
| 7153 | horse | (horse, horse) | (airplane, horse) | 19 |
| 6006 | airplane | (airplane, airplane) | (horse, airplane) | 12 |
| 341 | frog | (frog, frog) | (cat, frog) | 5 |
| 3879 | horse | (horse, horse) | (horse, airplane) | 8 |
| 3773 | bird | (bird, bird) | (dog, bird) | 2 |

## Saved Files
- `coverage_summary.png`
- `clean_disagreements/`
- `generated_inputs/`
