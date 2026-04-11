#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm.auto import tqdm


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
MODEL_NAME_LIST = ["resnet50_wo_aug", "resnet50_w_aug"]
MODEL_RANDOM_SEED_LIST = [2026, 2027]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Light-only DeepXplore-style differential testing for CIFAR-10 ResNet50 models"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--coverage-batch-size", type=int, default=128)
    parser.add_argument("--coverage-threshold", type=float, default=0.2)
    parser.add_argument("--coverage-batch-limit", type=int, default=None)
    parser.add_argument("--seed-limit", type=int, default=200)
    parser.add_argument("--grad-iterations", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=1 / 255)
    parser.add_argument("--max-perturbation", type=float, default=8 / 255)
    parser.add_argument("--coverage-weight", type=float, default=0.1)
    parser.add_argument("--generated-input-limit", type=int, default=5)
    parser.add_argument("--target-model-index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output-dir", type=str, default="results/test_light")
    parser.add_argument("--skip-save-figures", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cifar10_resnet50(random_seed):
    torch.manual_seed(random_seed)
    cifar10_resnet50_model = resnet50(weights=None)
    cifar10_resnet50_model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    cifar10_resnet50_model.maxpool = nn.Identity()
    cifar10_resnet50_model.fc = nn.Linear(
        in_features=cifar10_resnet50_model.fc.in_features,
        out_features=len(CLASS_NAMES),
    )
    if torch.cuda.is_available():
        cifar10_resnet50_model = cifar10_resnet50_model.cuda()
    return cifar10_resnet50_model


def load_trained_resnet50(model_name, random_seed):
    trained_resnet50_model = build_cifar10_resnet50(random_seed)
    checkpoint = torch.load(
        Path("ckpts") / f"{model_name}.pt",
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    trained_resnet50_model.load_state_dict(checkpoint["model_state_dict"])
    trained_resnet50_model.eval()
    return trained_resnet50_model, checkpoint


def preprocess_image(image_tensor):
    mean_tensor = image_tensor.new_tensor(CIFAR10_MEAN).view(3, 1, 1)
    std_tensor = image_tensor.new_tensor(CIFAR10_STD).view(3, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor


def deprocess_image(image_tensor):
    return image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def calculate_accuracy(output_logits, label_batch):
    predicted_label_batch = output_logits.argmax(dim=1)
    return (predicted_label_batch == label_batch).float().mean().item()


def evaluate_model(model_name, resnet50_model, data_loader, loss_function):
    loss_sum = 0.0
    accuracy_sum = 0.0
    batch_count = 0

    eval_progress_bar = tqdm(
        data_loader,
        desc=f"{model_name} eval",
        leave=False,
        dynamic_ncols=True,
    )

    resnet50_model.eval()
    with torch.no_grad():
        for input_batch, label_batch in eval_progress_bar:
            if torch.cuda.is_available():
                input_batch = input_batch.cuda(non_blocking=True)
                label_batch = label_batch.cuda(non_blocking=True)

            output_logits = resnet50_model(input_batch)
            batch_loss = loss_function(output_logits, label_batch)

            loss_sum += batch_loss.item()
            accuracy_sum += calculate_accuracy(output_logits, label_batch)
            batch_count += 1

            eval_progress_bar.set_postfix(
                loss=f"{loss_sum / batch_count:.4f}",
                acc=f"{accuracy_sum / batch_count:.4f}",
            )

    return loss_sum / batch_count, accuracy_sum / batch_count


def collect_prediction_tensor(model_name, resnet50_model, data_loader):
    predicted_label_batch_list = []
    true_label_batch_list = []

    prediction_progress_bar = tqdm(
        data_loader,
        desc=f"{model_name} predict",
        leave=False,
        dynamic_ncols=True,
    )

    resnet50_model.eval()
    with torch.no_grad():
        for input_batch, label_batch in prediction_progress_bar:
            if torch.cuda.is_available():
                input_batch = input_batch.cuda(non_blocking=True)

            output_logits = resnet50_model(input_batch)
            predicted_label_batch_list.append(output_logits.argmax(dim=1).cpu())
            true_label_batch_list.append(label_batch.cpu())

    return torch.cat(predicted_label_batch_list), torch.cat(true_label_batch_list)


def normalize_neuron_activation(neuron_activation_tensor):
    minimum_activation = neuron_activation_tensor.min(dim=1, keepdim=True).values
    maximum_activation = neuron_activation_tensor.max(dim=1, keepdim=True).values
    return (neuron_activation_tensor - minimum_activation) / (
        maximum_activation - minimum_activation + 1e-6
    )


def measure_neuron_coverage(model_name, resnet50_model, data_loader, activation_threshold, batch_limit):
    covered_neuron_dict = {}
    hook_handle_list = []

    for layer_name, layer_module in resnet50_model.named_modules():
        if isinstance(layer_module, nn.Conv2d):
            covered_neuron_dict[layer_name] = torch.zeros(layer_module.out_channels, dtype=torch.bool)

            def conv_hook(module, input_value, output_value, layer_name=layer_name):
                neuron_activation_tensor = output_value.detach().float().mean(dim=(2, 3)).cpu()
                normalized_neuron_activation_tensor = normalize_neuron_activation(neuron_activation_tensor)
                covered_neuron_dict[layer_name] |= (
                    normalized_neuron_activation_tensor > activation_threshold
                ).any(dim=0)

            hook_handle_list.append(layer_module.register_forward_hook(conv_hook))

        if isinstance(layer_module, nn.Linear):
            covered_neuron_dict[layer_name] = torch.zeros(layer_module.out_features, dtype=torch.bool)

            def linear_hook(module, input_value, output_value, layer_name=layer_name):
                neuron_activation_tensor = output_value.detach().float().cpu()
                normalized_neuron_activation_tensor = normalize_neuron_activation(neuron_activation_tensor)
                covered_neuron_dict[layer_name] |= (
                    normalized_neuron_activation_tensor > activation_threshold
                ).any(dim=0)

            hook_handle_list.append(layer_module.register_forward_hook(linear_hook))

    resnet50_model.eval()
    coverage_progress_bar = tqdm(
        data_loader,
        desc=f"{model_name} coverage",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch_index, (input_batch, _) in enumerate(coverage_progress_bar):
            if batch_limit is not None and batch_index >= batch_limit:
                break

            if torch.cuda.is_available():
                input_batch = input_batch.cuda(non_blocking=True)

            _ = resnet50_model(input_batch)

    for hook_handle in hook_handle_list:
        hook_handle.remove()

    covered_neuron_count = sum(
        covered_neuron_tensor.sum().item() for covered_neuron_tensor in covered_neuron_dict.values()
    )
    total_neuron_count = sum(covered_neuron_tensor.numel() for covered_neuron_tensor in covered_neuron_dict.values())
    neuron_coverage_ratio = covered_neuron_count / total_neuron_count

    return covered_neuron_dict, covered_neuron_count, total_neuron_count, neuron_coverage_ratio


def find_uncovered_neuron(covered_neuron_dict):
    for layer_name, covered_neuron_tensor in covered_neuron_dict.items():
        uncovered_neuron_index_tensor = torch.nonzero(~covered_neuron_tensor, as_tuple=False).view(-1)
        if uncovered_neuron_index_tensor.numel() > 0:
            return layer_name, uncovered_neuron_index_tensor[0].item()
    last_layer_name = list(covered_neuron_dict.keys())[-1]
    return last_layer_name, 0


def forward_with_target_neuron(resnet50_model, normalized_input_batch, target_layer_name, target_neuron_index):
    captured_output_dict = {}

    def save_output_hook(module, input_value, output_value):
        captured_output_dict["target_layer_output"] = output_value

    target_layer_module = dict(resnet50_model.named_modules())[target_layer_name]
    hook_handle = target_layer_module.register_forward_hook(save_output_hook)
    output_logits = resnet50_model(normalized_input_batch)
    hook_handle.remove()

    target_layer_output = captured_output_dict["target_layer_output"]
    target_neuron_activation = target_layer_output[:, target_neuron_index].mean()
    return output_logits, target_neuron_activation


def normalize_gradient(gradient_tensor):
    return gradient_tensor / (gradient_tensor.abs().mean() + 1e-8)


def apply_light_constraint(gradient_tensor):
    return gradient_tensor.mean() * torch.ones_like(gradient_tensor)


def predict_label_index_list(trained_model_list, image_tensor):
    normalized_input_batch = preprocess_image(image_tensor).unsqueeze(0)
    with torch.no_grad():
        return [
            trained_resnet50_model(normalized_input_batch).argmax(dim=1).item()
            for trained_resnet50_model in trained_model_list
        ]


def generate_disagreement_input(
    seed_image_tensor,
    trained_model_list,
    target_model_index,
    target_layer_name,
    target_neuron_index,
    grad_iterations,
    step_size,
    max_perturbation,
    coverage_weight,
):
    if torch.cuda.is_available():
        original_image_tensor = seed_image_tensor.clone().cuda()
    else:
        original_image_tensor = seed_image_tensor.clone()

    generated_image_tensor = original_image_tensor.clone()
    original_prediction_index_list = predict_label_index_list(trained_model_list, original_image_tensor)

    if original_prediction_index_list[0] != original_prediction_index_list[1]:
        return None

    original_prediction_index = original_prediction_index_list[0]
    other_model_index = 1 - target_model_index

    for iteration_index in range(grad_iterations):
        generated_image_tensor = generated_image_tensor.detach().requires_grad_(True)
        normalized_input_batch = preprocess_image(generated_image_tensor).unsqueeze(0)

        target_model_logits, target_neuron_activation = forward_with_target_neuron(
            trained_model_list[target_model_index],
            normalized_input_batch,
            target_layer_name,
            target_neuron_index,
        )
        other_model_logits = trained_model_list[other_model_index](normalized_input_batch)

        disagreement_loss = (
            -target_model_logits[0, original_prediction_index]
            + other_model_logits[0, original_prediction_index]
        )
        total_loss = disagreement_loss + coverage_weight * target_neuron_activation
        total_loss.backward()

        gradient_tensor = generated_image_tensor.grad.detach()
        normalized_gradient_tensor = normalize_gradient(gradient_tensor)
        light_gradient_tensor = apply_light_constraint(normalized_gradient_tensor)

        with torch.no_grad():
            generated_image_tensor = generated_image_tensor + step_size * light_gradient_tensor.sign()
            lower_bound_tensor = (original_image_tensor - max_perturbation).clamp(0.0, 1.0)
            upper_bound_tensor = (original_image_tensor + max_perturbation).clamp(0.0, 1.0)
            generated_image_tensor = torch.max(
                torch.min(generated_image_tensor, upper_bound_tensor),
                lower_bound_tensor,
            )
            generated_image_tensor = generated_image_tensor.clamp(0.0, 1.0)

        generated_prediction_index_list = predict_label_index_list(trained_model_list, generated_image_tensor)
        if generated_prediction_index_list[0] != generated_prediction_index_list[1]:
            return {
                "generated_image_tensor": generated_image_tensor.detach().cpu(),
                "original_image_tensor": original_image_tensor.detach().cpu(),
                "original_prediction_index_list": original_prediction_index_list,
                "generated_prediction_index_list": generated_prediction_index_list,
                "iteration_index": iteration_index + 1,
            }

    return None


def clone_covered_neuron_dict(covered_neuron_dict):
    return {
        layer_name: covered_neuron_tensor.clone()
        for layer_name, covered_neuron_tensor in covered_neuron_dict.items()
    }


def update_generated_input_coverage(resnet50_model, image_batch, covered_neuron_dict, activation_threshold):
    hook_handle_list = []

    for layer_name, layer_module in resnet50_model.named_modules():
        if isinstance(layer_module, nn.Conv2d):

            def conv_hook(module, input_value, output_value, layer_name=layer_name):
                neuron_activation_tensor = output_value.detach().float().mean(dim=(2, 3)).cpu()
                normalized_neuron_activation_tensor = normalize_neuron_activation(neuron_activation_tensor)
                covered_neuron_dict[layer_name] |= (
                    normalized_neuron_activation_tensor > activation_threshold
                ).any(dim=0)

            hook_handle_list.append(layer_module.register_forward_hook(conv_hook))

        if isinstance(layer_module, nn.Linear):

            def linear_hook(module, input_value, output_value, layer_name=layer_name):
                neuron_activation_tensor = output_value.detach().float().cpu()
                normalized_neuron_activation_tensor = normalize_neuron_activation(neuron_activation_tensor)
                covered_neuron_dict[layer_name] |= (
                    normalized_neuron_activation_tensor > activation_threshold
                ).any(dim=0)

            hook_handle_list.append(layer_module.register_forward_hook(linear_hook))

    resnet50_model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            image_batch = image_batch.cuda(non_blocking=True)
        normalized_image_batch = preprocess_image(image_batch)
        _ = resnet50_model(normalized_image_batch)

    for hook_handle in hook_handle_list:
        hook_handle.remove()


def save_generated_input_figure(output_path, generation_result):
    save_figure, save_axis_list = plt.subplots(1, 2, figsize=(6, 3))

    save_axis_list[0].imshow(deprocess_image(generation_result["original_image_tensor"]))
    save_axis_list[0].set_title(
        f"seed {generation_result['seed_index']}\n"
        f"true: {CLASS_NAMES[generation_result['true_label_index']]}\n"
        f"clean: {CLASS_NAMES[generation_result['original_prediction_index_list'][0]]}"
    )
    save_axis_list[0].axis("off")

    save_axis_list[1].imshow(deprocess_image(generation_result["generated_image_tensor"]))
    save_axis_list[1].set_title(
        f"iter: {generation_result['iteration_index']}\n"
        f"wo: {CLASS_NAMES[generation_result['generated_prediction_index_list'][0]]}\n"
        f"w: {CLASS_NAMES[generation_result['generated_prediction_index_list'][1]]}"
    )
    save_axis_list[1].axis("off")

    save_figure.tight_layout()
    save_figure.savefig(output_path, bbox_inches="tight")
    plt.close(save_figure)


def main():
    args = parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"output dir: {Path(args.output_dir).resolve()}")

    raw_test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    model_input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    model_test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=model_input_transform,
    )

    model_test_loader = DataLoader(
        model_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    coverage_test_loader = DataLoader(
        model_test_dataset,
        batch_size=args.coverage_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loss_function = nn.CrossEntropyLoss()
    trained_model_list = []
    model_summary_list = []

    for model_name, random_seed in zip(MODEL_NAME_LIST, MODEL_RANDOM_SEED_LIST):
        trained_resnet50_model, checkpoint = load_trained_resnet50(model_name, random_seed)
        trained_model_list.append(trained_resnet50_model)

        test_loss, test_accuracy = evaluate_model(
            model_name,
            trained_resnet50_model,
            model_test_loader,
            test_loss_function,
        )

        model_summary = {
            "model_name": model_name,
            "saved_epoch": int(checkpoint["epoch"]),
            "validation_accuracy": float(checkpoint["validation_accuracy"]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        }
        model_summary_list.append(model_summary)

        print(
            f"{model_name} | saved epoch {checkpoint['epoch']} | "
            f"val acc {checkpoint['validation_accuracy']:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_accuracy:.4f}"
        )

    predicted_label_index_tensor_for_wo_aug, true_label_index_tensor = collect_prediction_tensor(
        "resnet50_wo_aug",
        trained_model_list[0],
        model_test_loader,
    )
    predicted_label_index_tensor_for_w_aug, _ = collect_prediction_tensor(
        "resnet50_w_aug",
        trained_model_list[1],
        model_test_loader,
    )

    disagreement_index_list = torch.nonzero(
        predicted_label_index_tensor_for_wo_aug != predicted_label_index_tensor_for_w_aug,
        as_tuple=False,
    ).view(-1).tolist()
    clean_disagreement_rate = len(disagreement_index_list) / len(model_test_dataset)

    print(f"clean disagreement count: {len(disagreement_index_list)}")
    print(f"clean disagreement rate: {clean_disagreement_rate:.4f}")

    coverage_result_dict = {}
    for model_name, trained_resnet50_model in zip(MODEL_NAME_LIST, trained_model_list):
        covered_neuron_dict, covered_neuron_count, total_neuron_count, neuron_coverage_ratio = measure_neuron_coverage(
            model_name,
            trained_resnet50_model,
            coverage_test_loader,
            args.coverage_threshold,
            args.coverage_batch_limit,
        )
        coverage_result_dict[model_name] = {
            "covered_neuron_dict": covered_neuron_dict,
            "covered_neuron_count": covered_neuron_count,
            "total_neuron_count": total_neuron_count,
            "neuron_coverage_ratio": neuron_coverage_ratio,
        }
        print(
            f"{model_name} | covered neurons {covered_neuron_count}/{total_neuron_count} | "
            f"neuron coverage {neuron_coverage_ratio:.4f}"
        )

    average_neuron_coverage = sum(
        coverage_result_dict[model_name]["neuron_coverage_ratio"]
        for model_name in MODEL_NAME_LIST
    ) / len(MODEL_NAME_LIST)
    print(f"average neuron coverage: {average_neuron_coverage:.4f}")

    target_model_name = MODEL_NAME_LIST[args.target_model_index]
    target_layer_name, target_neuron_index = find_uncovered_neuron(
        coverage_result_dict[target_model_name]["covered_neuron_dict"]
    )
    agreement_index_list = torch.nonzero(
        predicted_label_index_tensor_for_wo_aug == predicted_label_index_tensor_for_w_aug,
        as_tuple=False,
    ).view(-1).tolist()

    print(f"target model: {target_model_name}")
    print(f"target neuron: {target_layer_name}[{target_neuron_index}]")

    generated_result_list = []
    generation_progress_bar = tqdm(
        agreement_index_list[: args.seed_limit],
        desc="generate disagreement",
        leave=False,
        dynamic_ncols=True,
    )

    for disagreement_seed_index in generation_progress_bar:
        seed_image_tensor, true_label_index = raw_test_dataset[disagreement_seed_index]
        generation_result = generate_disagreement_input(
            seed_image_tensor,
            trained_model_list,
            args.target_model_index,
            target_layer_name,
            target_neuron_index,
            args.grad_iterations,
            args.step_size,
            args.max_perturbation,
            args.coverage_weight,
        )

        if generation_result is None:
            continue

        generation_result["seed_index"] = disagreement_seed_index
        generation_result["true_label_index"] = true_label_index
        generated_result_list.append(generation_result)
        generation_progress_bar.set_postfix(found=len(generated_result_list))

        if len(generated_result_list) >= args.generated_input_limit:
            break

    print(f"generated disagreement count: {len(generated_result_list)}")

    if not args.skip_save_figures:
        for result_index, generation_result in enumerate(generated_result_list):
            save_generated_input_figure(
                Path(args.output_dir) / f"generated_{result_index:02d}.png",
                generation_result,
            )

    coverage_gain_summary_dict = {}
    generated_input_summary_list = []

    if len(generated_result_list) > 0:
        generated_image_batch = torch.stack(
            [generation_result["generated_image_tensor"] for generation_result in generated_result_list]
        )

        for model_name, trained_resnet50_model in zip(MODEL_NAME_LIST, trained_model_list):
            baseline_covered_neuron_dict = coverage_result_dict[model_name]["covered_neuron_dict"]
            updated_covered_neuron_dict = clone_covered_neuron_dict(baseline_covered_neuron_dict)

            update_generated_input_coverage(
                trained_resnet50_model,
                generated_image_batch,
                updated_covered_neuron_dict,
                args.coverage_threshold,
            )

            baseline_covered_neuron_count = coverage_result_dict[model_name]["covered_neuron_count"]
            total_neuron_count = coverage_result_dict[model_name]["total_neuron_count"]
            updated_covered_neuron_count = sum(
                covered_neuron_tensor.sum().item()
                for covered_neuron_tensor in updated_covered_neuron_dict.values()
            )
            updated_neuron_coverage_ratio = updated_covered_neuron_count / total_neuron_count
            added_neuron_count = updated_covered_neuron_count - baseline_covered_neuron_count

            coverage_gain_summary_dict[model_name] = {
                "baseline_covered_neuron_count": int(baseline_covered_neuron_count),
                "updated_covered_neuron_count": int(updated_covered_neuron_count),
                "total_neuron_count": int(total_neuron_count),
                "baseline_neuron_coverage_ratio": float(
                    coverage_result_dict[model_name]["neuron_coverage_ratio"]
                ),
                "updated_neuron_coverage_ratio": float(updated_neuron_coverage_ratio),
                "added_neuron_count": int(added_neuron_count),
            }

            print(
                f"{model_name} | baseline {baseline_covered_neuron_count}/{total_neuron_count} "
                f"({coverage_result_dict[model_name]['neuron_coverage_ratio']:.4f}) -> "
                f"generated {updated_covered_neuron_count}/{total_neuron_count} "
                f"({updated_neuron_coverage_ratio:.4f}) | +{added_neuron_count} neurons"
            )

        for generation_result in generated_result_list:
            generated_input_summary_list.append(
                {
                    "seed_index": int(generation_result["seed_index"]),
                    "true_label": CLASS_NAMES[generation_result["true_label_index"]],
                    "original_prediction_wo_aug": CLASS_NAMES[
                        generation_result["original_prediction_index_list"][0]
                    ],
                    "original_prediction_w_aug": CLASS_NAMES[
                        generation_result["original_prediction_index_list"][1]
                    ],
                    "generated_prediction_wo_aug": CLASS_NAMES[
                        generation_result["generated_prediction_index_list"][0]
                    ],
                    "generated_prediction_w_aug": CLASS_NAMES[
                        generation_result["generated_prediction_index_list"][1]
                    ],
                    "iteration_count": int(generation_result["iteration_index"]),
                }
            )

        for generated_input_summary in generated_input_summary_list:
            print(
                f"seed {generated_input_summary['seed_index']} | "
                f"true {generated_input_summary['true_label']} | "
                f"clean ({generated_input_summary['original_prediction_wo_aug']}, "
                f"{generated_input_summary['original_prediction_w_aug']}) -> "
                f"generated ({generated_input_summary['generated_prediction_wo_aug']}, "
                f"{generated_input_summary['generated_prediction_w_aug']}) | "
                f"iters {generated_input_summary['iteration_count']}"
            )

    summary_dict = {
        "transformation": "light",
        "model_summary_list": model_summary_list,
        "clean_disagreement_count": len(disagreement_index_list),
        "clean_disagreement_rate": clean_disagreement_rate,
        "average_neuron_coverage": average_neuron_coverage,
        "coverage_summary_dict": {
            model_name: {
                "covered_neuron_count": int(coverage_result_dict[model_name]["covered_neuron_count"]),
                "total_neuron_count": int(coverage_result_dict[model_name]["total_neuron_count"]),
                "neuron_coverage_ratio": float(coverage_result_dict[model_name]["neuron_coverage_ratio"]),
            }
            for model_name in MODEL_NAME_LIST
        },
        "generated_disagreement_count": len(generated_result_list),
        "coverage_gain_summary_dict": coverage_gain_summary_dict,
        "generated_input_summary_list": generated_input_summary_list,
    }

    with open(Path(args.output_dir) / "summary.json", "w") as file:
        json.dump(summary_dict, file, indent=2)

    print(f"saved: {(Path(args.output_dir) / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
