#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from utils import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CLASS_NAME_LIST,
    MODEL_NAME_LIST,
    MODEL_RANDOM_SEED_LIST,
    collect_prediction_tensor,
    constraint_black,
    constraint_light,
    constraint_occl,
    diverged,
    eval_model,
    fired,
    forward_layer,
    init_coverage_tables,
    load_trained_resnet50,
    neuron_covered,
    neuron_to_cover,
    normalize,
    predict_label,
    preprocess_image,
    save_clean_disagreement_figure,
    save_coverage_summary_figure,
    save_experiment_report,
    save_generated_input_figure,
    set_seed,
    update_coverage,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepXplore-style differential testing for CIFAR-10 ResNet50 models"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--coverage-batch-size", type=int, default=128)
    parser.add_argument("--coverage-batch-limit", type=int, default=None)
    parser.add_argument("--threshold", "--coverage-threshold", dest="threshold", type=float, default=0.2)
    parser.add_argument(
        "--transformation",
        type=str,
        default="all",
        choices=["all", "light", "occl", "black", "blackout"],
    )
    parser.add_argument("--weight-diff", type=float, default=5.0)
    parser.add_argument("--weight-nc", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=1 / 255)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--grad-iterations", type=int, default=20)
    parser.add_argument("--target-model", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-perturbation", type=float, default=100 / 255)
    parser.add_argument("--generated-input-limit", type=int, default=5)
    parser.add_argument("--clean-figure-limit", type=int, default=5)
    parser.add_argument("--occl-start-row", type=int, default=10)
    parser.add_argument("--occl-start-col", type=int, default=10)
    parser.add_argument("--occl-height", type=int, default=8)
    parser.add_argument("--occl-width", type=int, default=8)
    parser.add_argument("--black-height", type=int, default=6)
    parser.add_argument("--black-width", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-save-figures", action="store_true")
    return parser.parse_args()


def resolve_transformation_name_list(requested_transformation_name):
    if requested_transformation_name == "all":
        return ["light", "occl", "black"]
    if requested_transformation_name == "blackout":
        return ["black"]
    return [requested_transformation_name]


def clear_file_if_exists(path):
    if path.exists():
        path.unlink()


def clear_matching_files(directory, pattern):
    if not directory.exists():
        return
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()


def prepare_transformation_output_dir(transformation_output_dir):
    generated_input_dir = transformation_output_dir / "generated_inputs"
    transformation_output_dir.mkdir(parents=True, exist_ok=True)
    generated_input_dir.mkdir(parents=True, exist_ok=True)

    clear_matching_files(generated_input_dir, "*.png")
    clear_file_if_exists(transformation_output_dir / "generated_inputs_overview.png")
    clear_file_if_exists(transformation_output_dir / "coverage_summary.png")
    clear_file_if_exists(transformation_output_dir / "report.md")
    clear_file_if_exists(transformation_output_dir / "summary.json")

    return generated_input_dir


def run_generation_for_transformation(
    args,
    transformation_name,
    transformation_output_dir,
    raw_test_dataset,
    model1,
    model2,
    baseline_model_layer_dict1,
    baseline_model_layer_dict2,
    agreement_index_list,
    model_summary_list,
    clean_disagreement_count,
    clean_disagreement_rate,
    saved_clean_figure_count,
):
    generated_input_dir = prepare_transformation_output_dir(transformation_output_dir)
    coverage_summary_path = transformation_output_dir / "coverage_summary.png"
    report_path = transformation_output_dir / "report.md"

    model_layer_dict1 = baseline_model_layer_dict1.copy()
    model_layer_dict2 = baseline_model_layer_dict2.copy()

    baseline_covered_neurons1, baseline_total_neurons1, baseline_neuron_coverage1 = neuron_covered(model_layer_dict1)
    baseline_covered_neurons2, baseline_total_neurons2, baseline_neuron_coverage2 = neuron_covered(model_layer_dict2)

    generated_result_list = []
    generation_progress_bar = tqdm(
        agreement_index_list[: args.seeds],
        desc=f"gen_diff[{transformation_name}]",
        leave=False,
        dynamic_ncols=True,
    )

    print()
    print(f"[{transformation_name}] target model: {MODEL_NAME_LIST[args.target_model]}")

    for seed_index in generation_progress_bar:
        gen_img, true_label = raw_test_dataset[seed_index]
        if torch.cuda.is_available():
            gen_img = gen_img.cuda()

        orig_img = gen_img.clone()
        label1 = predict_label(model1, gen_img)
        label2 = predict_label(model2, gen_img)
        orig_label = label1

        layer_name1, index1 = neuron_to_cover(model_layer_dict1)
        layer_name2, index2 = neuron_to_cover(model_layer_dict2)

        for iters in range(args.grad_iterations):
            gen_img = gen_img.detach().requires_grad_(True)
            input_data = preprocess_image(gen_img).unsqueeze(0)

            output1, layer_output1 = forward_layer(model1, input_data, layer_name1)
            output2, layer_output2 = forward_layer(model2, input_data, layer_name2)

            if args.target_model == 0:
                loss1 = -args.weight_diff * torch.mean(output1[..., orig_label])
                loss2 = torch.mean(output2[..., orig_label])
            else:
                loss1 = torch.mean(output1[..., orig_label])
                loss2 = -args.weight_diff * torch.mean(output2[..., orig_label])

            if layer_output1.dim() == 4:
                loss1_neuron = torch.mean(layer_output1[:, index1, :, :])
            else:
                loss1_neuron = torch.mean(layer_output1[..., index1])

            if layer_output2.dim() == 4:
                loss2_neuron = torch.mean(layer_output2[:, index2, :, :])
            else:
                loss2_neuron = torch.mean(layer_output2[..., index2])

            final_loss = (loss1 + loss2) + args.weight_nc * (loss1_neuron + loss2_neuron)
            final_loss.backward()
            grads = normalize(gen_img.grad.detach())

            if transformation_name == "light":
                grads = constraint_light(grads)
            elif transformation_name == "occl":
                grads = constraint_occl(
                    grads,
                    args.occl_start_row,
                    args.occl_start_col,
                    args.occl_height,
                    args.occl_width,
                )
            else:
                grads = constraint_black(
                    grads,
                    args.black_height,
                    args.black_width,
                )

            with torch.no_grad():
                gen_img = gen_img + grads * args.step
                lower_bound = (orig_img - args.max_perturbation).clamp(0.0, 1.0)
                upper_bound = (orig_img + args.max_perturbation).clamp(0.0, 1.0)
                gen_img = torch.max(torch.min(gen_img, upper_bound), lower_bound)
                gen_img = gen_img.clamp(0.0, 1.0)

            predictions1 = predict_label(model1, gen_img)
            predictions2 = predict_label(model2, gen_img)

            if diverged(predictions1, predictions2, args.target_model):
                input_data = preprocess_image(gen_img).unsqueeze(0)
                update_coverage(input_data, model1, model_layer_dict1, args.threshold)
                update_coverage(input_data, model2, model_layer_dict2, args.threshold)

                covered_neurons1, total_neurons1, neuron_coverage1 = neuron_covered(model_layer_dict1)
                covered_neurons2, total_neurons2, neuron_coverage2 = neuron_covered(model_layer_dict2)
                averaged_nc = (covered_neurons1 + covered_neurons2) / float(total_neurons1 + total_neurons2)

                generated_result_list.append(
                    {
                        "seed_index": int(seed_index),
                        "true_label": int(true_label),
                        "orig_img": orig_img.detach().cpu(),
                        "gen_img": gen_img.detach().cpu(),
                        "orig_label": int(orig_label),
                        "label1": int(label1),
                        "label2": int(label2),
                        "predictions1": int(predictions1),
                        "predictions2": int(predictions2),
                        "iters": int(iters + 1),
                        "layer_name1": layer_name1,
                        "index1": int(index1),
                        "layer_name2": layer_name2,
                        "index2": int(index2),
                        "fired1": bool(fired(model1, layer_name1, index1, input_data, args.threshold)),
                        "fired2": bool(fired(model2, layer_name2, index2, input_data, args.threshold)),
                        "transformation": transformation_name,
                        "type": "generated",
                    }
                )

                generation_progress_bar.set_postfix(
                    found=len(generated_result_list),
                    nc=f"{averaged_nc:.4f}",
                )
                break

        if len(generated_result_list) >= args.generated_input_limit:
            break

    print(f"[{transformation_name}] generated disagreement count: {len(generated_result_list)}")
    if len(generated_result_list) == 0:
        print(f"[{transformation_name}] generated disagreement example not found")

    saved_generated_figure_count = 0
    if not args.skip_save_figures:
        saved_generated_figure_count = len(generated_result_list)
        for result_index, generation_result in enumerate(generated_result_list):
            save_generated_input_figure(
                generated_input_dir
                / f"{transformation_name}_{generation_result['predictions1']}_{generation_result['predictions2']}_{result_index:02d}.png",
                generation_result,
            )

    final_covered_neurons1, final_total_neurons1, final_neuron_coverage1 = neuron_covered(model_layer_dict1)
    final_covered_neurons2, final_total_neurons2, final_neuron_coverage2 = neuron_covered(model_layer_dict2)
    final_average_neuron_coverage = (final_covered_neurons1 + final_covered_neurons2) / float(
        final_total_neurons1 + final_total_neurons2
    )

    coverage_gain_summary_dict = {
        MODEL_NAME_LIST[0]: {
            "baseline_covered_neuron_count": int(baseline_covered_neurons1),
            "updated_covered_neuron_count": int(final_covered_neurons1),
            "total_neuron_count": int(final_total_neurons1),
            "baseline_neuron_coverage_ratio": float(baseline_neuron_coverage1),
            "updated_neuron_coverage_ratio": float(final_neuron_coverage1),
            "added_neuron_count": int(final_covered_neurons1 - baseline_covered_neurons1),
        },
        MODEL_NAME_LIST[1]: {
            "baseline_covered_neuron_count": int(baseline_covered_neurons2),
            "updated_covered_neuron_count": int(final_covered_neurons2),
            "total_neuron_count": int(final_total_neurons2),
            "baseline_neuron_coverage_ratio": float(baseline_neuron_coverage2),
            "updated_neuron_coverage_ratio": float(final_neuron_coverage2),
            "added_neuron_count": int(final_covered_neurons2 - baseline_covered_neurons2),
        },
    }

    print(
        f"[{transformation_name}] {MODEL_NAME_LIST[0]} | baseline {baseline_covered_neurons1}/{baseline_total_neurons1} "
        f"({baseline_neuron_coverage1:.4f}) -> generated {final_covered_neurons1}/{final_total_neurons1} "
        f"({final_neuron_coverage1:.4f}) | +{final_covered_neurons1 - baseline_covered_neurons1} neurons"
    )
    print(
        f"[{transformation_name}] {MODEL_NAME_LIST[1]} | baseline {baseline_covered_neurons2}/{baseline_total_neurons2} "
        f"({baseline_neuron_coverage2:.4f}) -> generated {final_covered_neurons2}/{final_total_neurons2} "
        f"({final_neuron_coverage2:.4f}) | +{final_covered_neurons2 - baseline_covered_neurons2} neurons"
    )

    generated_input_summary_list = []
    for generation_result in generated_result_list:
        generated_input_summary = {
            "seed_index": generation_result["seed_index"],
            "true_label": CLASS_NAME_LIST[generation_result["true_label"]],
            "original_prediction_wo_aug": CLASS_NAME_LIST[generation_result["label1"]],
            "original_prediction_w_aug": CLASS_NAME_LIST[generation_result["label2"]],
            "generated_prediction_wo_aug": CLASS_NAME_LIST[generation_result["predictions1"]],
            "generated_prediction_w_aug": CLASS_NAME_LIST[generation_result["predictions2"]],
            "iteration_count": generation_result["iters"],
            "target_layer_for_wo_aug": generation_result["layer_name1"],
            "target_neuron_for_wo_aug": generation_result["index1"],
            "target_layer_for_w_aug": generation_result["layer_name2"],
            "target_neuron_for_w_aug": generation_result["index2"],
            "target_neuron_fired_for_wo_aug": generation_result["fired1"],
            "target_neuron_fired_for_w_aug": generation_result["fired2"],
            "transformation": generation_result["transformation"],
        }
        generated_input_summary_list.append(generated_input_summary)

        print(
            f"[{transformation_name}] seed {generated_input_summary['seed_index']} | "
            f"true {generated_input_summary['true_label']} | "
            f"clean ({generated_input_summary['original_prediction_wo_aug']}, "
            f"{generated_input_summary['original_prediction_w_aug']}) -> "
            f"generated ({generated_input_summary['generated_prediction_wo_aug']}, "
            f"{generated_input_summary['generated_prediction_w_aug']}) | "
            f"iters {generated_input_summary['iteration_count']}"
        )

    number_of_disagreement_inducing_inputs_found = len(generated_result_list)
    saved_suspicious_visualization_count = saved_clean_figure_count + saved_generated_figure_count

    print(
        f"[{transformation_name}] number of disagreement inducing inputs found: "
        f"{number_of_disagreement_inducing_inputs_found}"
    )
    print(f"[{transformation_name}] neuron coverage achieved: {final_average_neuron_coverage:.4f}")

    if not args.skip_save_figures:
        save_coverage_summary_figure(
            coverage_summary_path,
            transformation_name,
            MODEL_NAME_LIST[args.target_model],
            args.threshold,
            number_of_disagreement_inducing_inputs_found,
            final_average_neuron_coverage,
            coverage_gain_summary_dict,
        )

    save_experiment_report(
        report_path,
        transformation_name,
        MODEL_NAME_LIST[args.target_model],
        args.threshold,
        model_summary_list,
        clean_disagreement_count,
        clean_disagreement_rate,
        number_of_disagreement_inducing_inputs_found,
        final_average_neuron_coverage,
        coverage_gain_summary_dict,
        generated_input_summary_list,
    )

    if not args.skip_save_figures:
        print(f"[{transformation_name}] saved: {coverage_summary_path.resolve()}")
    print(f"[{transformation_name}] saved: {report_path.resolve()}")

    return {
        "transformation": transformation_name,
        "number_of_disagreement_inducing_inputs_found": number_of_disagreement_inducing_inputs_found,
        "final_average_neuron_coverage": final_average_neuron_coverage,
        "coverage_gain_summary_dict": coverage_gain_summary_dict,
        "report_path": report_path,
        "coverage_summary_path": coverage_summary_path,
    }


def main():
    args = parse_args()
    requested_transformation_name_list = resolve_transformation_name_list(args.transformation)
    run_multiple_transformations = len(requested_transformation_name_list) > 1

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    clean_disagreement_dir = output_dir / "clean_disagreements"
    legacy_generated_input_dir = output_dir / "generated_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_disagreement_dir.mkdir(parents=True, exist_ok=True)

    clear_matching_files(clean_disagreement_dir, "*.png")
    clear_file_if_exists(output_dir / "summary.json")
    clear_file_if_exists(output_dir / "experiment_overview.md")
    if run_multiple_transformations:
        clear_matching_files(legacy_generated_input_dir, "*.png")
        clear_file_if_exists(output_dir / "report.md")
        clear_file_if_exists(output_dir / "coverage_summary.png")
        clear_file_if_exists(output_dir / "generated_inputs_overview.png")

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"transformation request: {', '.join(requested_transformation_name_list)}")
    print(f"output dir: {output_dir.resolve()}")

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

        test_loss, test_accuracy = eval_model(
            model_name,
            trained_resnet50_model,
            model_test_loader,
            test_loss_function,
        )

        model_summary_list.append(
            {
                "model_name": model_name,
                "saved_epoch": int(checkpoint["epoch"]),
                "validation_accuracy": float(checkpoint["validation_accuracy"]),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
            }
        )

        print(
            f"{model_name} | saved epoch {checkpoint['epoch']} | "
            f"val acc {checkpoint['validation_accuracy']:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_accuracy:.4f}"
        )

    predicted_label_index_tensor_for_wo_aug, _ = collect_prediction_tensor(
        MODEL_NAME_LIST[0],
        trained_model_list[0],
        model_test_loader,
    )
    predicted_label_index_tensor_for_w_aug, _ = collect_prediction_tensor(
        MODEL_NAME_LIST[1],
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

    saved_clean_figure_count = 0
    if not args.skip_save_figures:
        saved_clean_figure_count = min(len(disagreement_index_list), args.clean_figure_limit)
        for result_index, disagreement_index in enumerate(disagreement_index_list[: args.clean_figure_limit]):
            image_tensor, true_label_index = raw_test_dataset[disagreement_index]
            save_clean_disagreement_figure(
                clean_disagreement_dir / f"clean_disagreement_{result_index:02d}.png",
                image_tensor,
                true_label_index,
                predicted_label_index_tensor_for_wo_aug[disagreement_index].item(),
                predicted_label_index_tensor_for_w_aug[disagreement_index].item(),
            )

    model1 = trained_model_list[0]
    model2 = trained_model_list[1]
    baseline_model_layer_dict1, baseline_model_layer_dict2 = init_coverage_tables(model1, model2)

    coverage_progress_bar = tqdm(
        coverage_test_loader,
        desc="update coverage",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_index, (input_batch, _) in enumerate(coverage_progress_bar):
        if args.coverage_batch_limit is not None and batch_index >= args.coverage_batch_limit:
            break

        if torch.cuda.is_available():
            input_batch = input_batch.cuda(non_blocking=True)

        update_coverage(input_batch, model1, baseline_model_layer_dict1, args.threshold)
        update_coverage(input_batch, model2, baseline_model_layer_dict2, args.threshold)

    baseline_covered_neurons1, baseline_total_neurons1, baseline_neuron_coverage1 = neuron_covered(
        baseline_model_layer_dict1
    )
    baseline_covered_neurons2, baseline_total_neurons2, baseline_neuron_coverage2 = neuron_covered(
        baseline_model_layer_dict2
    )
    baseline_average_neuron_coverage = (baseline_covered_neurons1 + baseline_covered_neurons2) / float(
        baseline_total_neurons1 + baseline_total_neurons2
    )

    print(
        f"{MODEL_NAME_LIST[0]} | covered neurons {baseline_covered_neurons1}/{baseline_total_neurons1} | "
        f"neuron coverage {baseline_neuron_coverage1:.4f}"
    )
    print(
        f"{MODEL_NAME_LIST[1]} | covered neurons {baseline_covered_neurons2}/{baseline_total_neurons2} | "
        f"neuron coverage {baseline_neuron_coverage2:.4f}"
    )
    print(f"average neuron coverage: {baseline_average_neuron_coverage:.4f}")

    agreement_index_list = torch.nonzero(
        predicted_label_index_tensor_for_wo_aug == predicted_label_index_tensor_for_w_aug,
        as_tuple=False,
    ).view(-1).tolist()
    random.Random(args.seed).shuffle(agreement_index_list)

    for transformation_index, transformation_name in enumerate(requested_transformation_name_list):
        set_seed(args.seed + transformation_index)
        transformation_output_dir = output_dir / transformation_name if run_multiple_transformations else output_dir

        run_generation_for_transformation(
            args,
            transformation_name,
            transformation_output_dir,
            raw_test_dataset,
            model1,
            model2,
            baseline_model_layer_dict1,
            baseline_model_layer_dict2,
            agreement_index_list,
            model_summary_list,
            len(disagreement_index_list),
            clean_disagreement_rate,
            saved_clean_figure_count,
        )


if __name__ == "__main__":
    main()
