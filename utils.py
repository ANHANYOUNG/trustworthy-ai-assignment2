import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from tqdm.auto import tqdm


CLASS_NAME_LIST = [
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
COVERAGE_LAYER_TYPE_TUPLE = (
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.AdaptiveAvgPool2d,
    nn.Identity,
)
COVERAGE_LAYER_SPEC_CACHE = {}

# ============================================================
# Experiment setup and model construction
# ============================================================
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
        out_features=len(CLASS_NAME_LIST),
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

# ============================================================
# Input preprocessing and tensor utilities
# ============================================================
def preprocess_image(image_tensor):
    mean_tensor = image_tensor.new_tensor(CIFAR10_MEAN).view(3, 1, 1)
    std_tensor = image_tensor.new_tensor(CIFAR10_STD).view(3, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor

def deprocess_image(image_tensor):
    return image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()

def normalize(input_tensor):
    return input_tensor / (torch.sqrt(torch.mean(torch.square(input_tensor))) + 1e-5)

def scale(intermediate_layer_output, rmax=1, rmin=0):
    minimum_value = intermediate_layer_output.min()
    maximum_value = intermediate_layer_output.max()
    if (maximum_value - minimum_value).abs().item() < 1e-8:
        return torch.zeros_like(intermediate_layer_output) + rmin
    scaled_output = (intermediate_layer_output - minimum_value) / (maximum_value - minimum_value)
    return scaled_output * (rmax - rmin) + rmin

# ============================================================
# Evaluation and prediction utilities
# ============================================================
def calculate_accuracy(output_logits, label_batch):
    predicted_label_batch = output_logits.argmax(dim=1)
    return (predicted_label_batch == label_batch).float().mean().item()

def eval_model(model_name, resnet50_model, data_loader, loss_function):
    loss_sum = 0.0
    accuracy_sum = 0.0
    batch_count = 0

    evaluation_progress_bar = tqdm(
        data_loader,
        desc=f"{model_name} eval",
        leave=False,
        dynamic_ncols=True,
    )

    resnet50_model.eval()
    with torch.no_grad():
        for input_batch, label_batch in evaluation_progress_bar:
            if torch.cuda.is_available():
                input_batch = input_batch.cuda(non_blocking=True)
                label_batch = label_batch.cuda(non_blocking=True)

            output_logits = resnet50_model(input_batch)
            batch_loss = loss_function(output_logits, label_batch)

            loss_sum += batch_loss.item()
            accuracy_sum += calculate_accuracy(output_logits, label_batch)
            batch_count += 1

            evaluation_progress_bar.set_postfix(
                loss=f"{loss_sum / batch_count:.4f}",
                acc=f"{accuracy_sum / batch_count:.4f}",
            )

    return loss_sum / batch_count, accuracy_sum / batch_count

def collect_prediction_tensor(model_name, resnet50_model, data_loader):
    predicted_label_index_tensor_list = []
    true_label_index_tensor_list = []

    prediction_progress_bar = tqdm(
        data_loader,
        desc=f"{model_name} predict",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for input_batch, label_batch in prediction_progress_bar:
            if torch.cuda.is_available():
                input_batch = input_batch.cuda(non_blocking=True)

            output_logits = resnet50_model(input_batch)
            predicted_label_index_tensor_list.append(output_logits.argmax(dim=1).cpu())
            true_label_index_tensor_list.append(label_batch.cpu())

    return torch.cat(predicted_label_index_tensor_list), torch.cat(true_label_index_tensor_list)

def predict_label(model, image_tensor):
    model.eval()
    with torch.no_grad():
        return model(preprocess_image(image_tensor).unsqueeze(0)).argmax(dim=1).item()

def diverged(predictions1, predictions2, target_model):
    return predictions1 != predictions2

# ============================================================
# Coverage layer naming and specification utilities
# ============================================================
def format_coverage_layer_name(layer_name, call_index):
    if call_index == 1:
        return layer_name
    return f"{layer_name}#{call_index}"

def parse_coverage_layer_name(coverage_layer_name):
    if "#" not in coverage_layer_name:
        return coverage_layer_name, 1
    layer_name, call_index = coverage_layer_name.rsplit("#", 1)
    return layer_name, int(call_index)

def build_coverage_layer_spec(model):
    layer_spec_list = []
    hook_handle_list = []
    layer_call_count_dict = defaultdict(int)

    for layer_name, layer_module in model.named_modules():
        if isinstance(layer_module, COVERAGE_LAYER_TYPE_TUPLE):

            def save_output(module, input_value, output_value, layer_name=layer_name):
                layer_call_count_dict[layer_name] += 1
                call_index = layer_call_count_dict[layer_name]
                coverage_layer_name = format_coverage_layer_name(layer_name, call_index)
                neuron_count = output_value.shape[1] if output_value.dim() >= 2 else output_value.shape[0]
                layer_spec_list.append((coverage_layer_name, neuron_count))

            hook_handle_list.append(layer_module.register_forward_hook(save_output))

    model.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device)
        _ = model(dummy_input)

    for hook_handle in hook_handle_list:
        hook_handle.remove()

    return layer_spec_list

def get_coverage_layer_spec(model):
    model_key = id(model)
    if model_key not in COVERAGE_LAYER_SPEC_CACHE:
        COVERAGE_LAYER_SPEC_CACHE[model_key] = build_coverage_layer_spec(model)
    return COVERAGE_LAYER_SPEC_CACHE[model_key]

# ============================================================
# Neuron coverage bookkeeping and querying
# ============================================================
def init_dict(model, model_layer_dict):
    for coverage_layer_name, neuron_count in get_coverage_layer_spec(model):
        for index in range(neuron_count):
            model_layer_dict[(coverage_layer_name, index)] = False

def init_coverage_tables(model1, model2):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    return model_layer_dict1, model_layer_dict2

def neuron_to_cover(model_layer_dict):
    uncovered_neuron_list = [
        (layer_name, index)
        for (layer_name, index), covered in model_layer_dict.items()
        if not covered
    ]
    if uncovered_neuron_list:
        return random.choice(uncovered_neuron_list)
    return random.choice(list(model_layer_dict.keys()))

def neuron_covered(model_layer_dict):
    covered_neuron_count = len([covered for covered in model_layer_dict.values() if covered])
    total_neuron_count = len(model_layer_dict)
    neuron_coverage_ratio = covered_neuron_count / float(total_neuron_count)
    return covered_neuron_count, total_neuron_count, neuron_coverage_ratio

def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_output_dict = {}
    hook_handle_list = []
    layer_call_count_dict = defaultdict(int)

    for layer_name, layer_module in model.named_modules():
        if isinstance(layer_module, COVERAGE_LAYER_TYPE_TUPLE):

            def save_output(module, input_value, output_value, layer_name=layer_name):
                layer_call_count_dict[layer_name] += 1
                call_index = layer_call_count_dict[layer_name]
                coverage_layer_name = format_coverage_layer_name(layer_name, call_index)
                layer_output_dict[coverage_layer_name] = output_value.detach()

            hook_handle_list.append(layer_module.register_forward_hook(save_output))

    model.eval()
    with torch.no_grad():
        _ = model(input_data)

    for hook_handle in hook_handle_list:
        hook_handle.remove()

    for layer_name, layer_output in layer_output_dict.items():
        scaled_output = scale(layer_output[0].float())
        for neuron_index in range(scaled_output.shape[0]):
            if scaled_output[neuron_index].mean().item() > threshold and not model_layer_dict[(layer_name, neuron_index)]:
                model_layer_dict[(layer_name, neuron_index)] = True

def fired(model, layer_name, index, input_data, threshold=0):
    layer_output_dict = {}
    actual_layer_name, target_call_index = parse_coverage_layer_name(layer_name)
    layer_call_count_dict = defaultdict(int)

    def save_output(module, input_value, output_value):
        layer_call_count_dict[actual_layer_name] += 1
        if layer_call_count_dict[actual_layer_name] == target_call_index:
            layer_output_dict["layer_output"] = output_value.detach()

    layer_module = dict(model.named_modules())[actual_layer_name]
    hook_handle = layer_module.register_forward_hook(save_output)
    model.eval()
    with torch.no_grad():
        _ = model(input_data)
    hook_handle.remove()

    scaled_output = scale(layer_output_dict["layer_output"][0].float())
    return scaled_output[index].mean().item() > threshold

def forward_layer(model, input_data, layer_name):
    layer_output_dict = {}
    actual_layer_name, target_call_index = parse_coverage_layer_name(layer_name)
    layer_call_count_dict = defaultdict(int)

    def save_output(module, input_value, output_value):
        layer_call_count_dict[actual_layer_name] += 1
        if layer_call_count_dict[actual_layer_name] == target_call_index:
            layer_output_dict["layer_output"] = output_value

    layer_module = dict(model.named_modules())[actual_layer_name]
    hook_handle = layer_module.register_forward_hook(save_output)
    output = model(input_data)
    hook_handle.remove()
    return output, layer_output_dict["layer_output"]

# ============================================================
# Input transformation constraints
# ============================================================
def constraint_occl(gradients, start_row, start_col, height, width):
    new_gradients = torch.zeros_like(gradients)
    new_gradients[:, start_row : start_row + height, start_col : start_col + width] = gradients[
        :,
        start_row : start_row + height,
        start_col : start_col + width,
    ]
    return new_gradients

def constraint_light(gradients):
    return gradients.mean() * torch.ones_like(gradients)

def constraint_black(gradients, height=6, width=6):
    start_row = random.randint(0, gradients.shape[1] - height)
    start_col = random.randint(0, gradients.shape[2] - width)
    new_gradients = torch.zeros_like(gradients)
    patch = gradients[:, start_row : start_row + height, start_col : start_col + width]
    if patch.mean().item() < 0:
        new_gradients[:, start_row : start_row + height, start_col : start_col + width] = -torch.ones_like(
            patch
        )
    return new_gradients

# ============================================================
# Visualization and result saving
# ============================================================
def build_diff_visualization(original_image_tensor, generated_image_tensor):
    original_image = deprocess_image(original_image_tensor)
    generated_image = deprocess_image(generated_image_tensor)
    diff_image = np.abs(generated_image - original_image)
    diff_maximum = float(diff_image.max())

    if diff_maximum > 1e-8:
        diff_visualization = diff_image / diff_maximum
    else:
        diff_visualization = diff_image

    return original_image, generated_image, diff_visualization, float(diff_image.mean()), diff_maximum

def save_clean_disagreement_figure(
    output_path,
    image_tensor,
    true_label_index,
    predicted_label_index_for_wo_aug,
    predicted_label_index_for_w_aug,
):
    save_figure, save_axis = plt.subplots(1, 1, figsize=(3, 3))
    save_axis.imshow(deprocess_image(image_tensor))
    save_axis.set_title(
        f"GT: {CLASS_NAME_LIST[true_label_index]}\n"
        f"wo: {CLASS_NAME_LIST[predicted_label_index_for_wo_aug]}\n"
        f"w: {CLASS_NAME_LIST[predicted_label_index_for_w_aug]}"
    )
    save_axis.axis("off")
    save_figure.tight_layout()
    save_figure.savefig(output_path, bbox_inches="tight")
    plt.close(save_figure)

def save_generated_input_figure(output_path, generation_result):
    original_image, generated_image, diff_visualization, diff_mean, diff_maximum = build_diff_visualization(
        generation_result["orig_img"],
        generation_result["gen_img"],
    )

    save_figure, save_axis_list = plt.subplots(1, 3, figsize=(9, 3))

    save_axis_list[0].imshow(original_image)
    save_axis_list[0].set_title(
        f"seed {generation_result['seed_index']}\n"
        f"true: {CLASS_NAME_LIST[generation_result['true_label']]}\n"
        f"wo: {CLASS_NAME_LIST[generation_result['label1']]} | "
        f"w: {CLASS_NAME_LIST[generation_result['label2']]}"
    )
    save_axis_list[0].axis("off")

    save_axis_list[1].imshow(generated_image)
    save_axis_list[1].set_title(
        f"iter: {generation_result['iters']}\n"
        f"wo: {CLASS_NAME_LIST[generation_result['predictions1']]}\n"
        f"w: {CLASS_NAME_LIST[generation_result['predictions2']]}"
    )
    save_axis_list[1].axis("off")

    save_axis_list[2].imshow(diff_visualization)
    save_axis_list[2].set_title(
        f"abs diff / max\n"
        f"mean: {diff_mean:.4f} | max: {diff_maximum:.4f}"
    )
    save_axis_list[2].axis("off")

    save_figure.tight_layout()
    save_figure.savefig(output_path, bbox_inches="tight")
    plt.close(save_figure)

def save_generated_input_overview(output_path, generated_result_list):
    if len(generated_result_list) == 0:
        return

    save_figure, axis_grid = plt.subplots(len(generated_result_list), 3, figsize=(12, 3 * len(generated_result_list)))
    if len(generated_result_list) == 1:
        axis_grid = np.expand_dims(axis_grid, axis=0)

    for result_index, generation_result in enumerate(generated_result_list):
        original_axis = axis_grid[result_index, 0]
        generated_axis = axis_grid[result_index, 1]
        diff_axis = axis_grid[result_index, 2]

        original_image, generated_image, diff_visualization, diff_mean, diff_maximum = build_diff_visualization(
            generation_result["orig_img"],
            generation_result["gen_img"],
        )

        original_axis.imshow(original_image)
        original_axis.set_title(
            f"seed {generation_result['seed_index']}\n"
            f"true: {CLASS_NAME_LIST[generation_result['true_label']]}\n"
            f"wo: {CLASS_NAME_LIST[generation_result['label1']]} | "
            f"w: {CLASS_NAME_LIST[generation_result['label2']]}"
        )
        original_axis.axis("off")

        generated_axis.imshow(generated_image)
        generated_axis.set_title(
            f"iter: {generation_result['iters']}\n"
            f"wo: {CLASS_NAME_LIST[generation_result['predictions1']]}\n"
            f"w: {CLASS_NAME_LIST[generation_result['predictions2']]}"
        )
        generated_axis.axis("off")

        diff_axis.imshow(diff_visualization)
        diff_axis.set_title(
            f"abs diff / max\n"
            f"mean: {diff_mean:.4f} | max: {diff_maximum:.4f}"
        )
        diff_axis.axis("off")

    save_figure.tight_layout()
    save_figure.savefig(output_path, bbox_inches="tight")
    plt.close(save_figure)

def save_coverage_summary_figure(
    output_path,
    transformation,
    target_model_name,
    coverage_threshold,
    number_of_disagreement_inducing_inputs_found,
    final_average_neuron_coverage,
    coverage_gain_summary_dict,
):
    transformation_display = transformation.replace("_", " ").strip()
    if transformation_display.islower():
        transformation_display = transformation_display.title()

    figure = plt.figure(figsize=(11.5, 6.4), facecolor="white")
    grid_spec = figure.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.9, 1.35, 2.75],
        left=0.06,
        right=0.94,
        top=0.93,
        bottom=0.11,
        hspace=0.22,
    )

    title_axis = figure.add_subplot(grid_spec[0])
    title_axis.set_xlim(0.0, 1.0)
    title_axis.set_ylim(0.0, 1.0)
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.88,
        f"Transformation: {transformation_display}",
        fontsize=22,
        fontweight="bold",
        color="#1f1f1f",
        va="top",
    )
    title_axis.text(
        0.0,
        0.34,
        "Coverage Summary",
        fontsize=12,
        color="#6a6a6a",
        va="top",
    )
    title_axis.plot([0.0, 1.0], [0.04, 0.04], color="#e2e2e2", linewidth=1.0)

    settings_axis = figure.add_subplot(grid_spec[1])
    settings_axis.axis("off")
    settings_axis.text(
        0.0,
        1.02,
        "Settings",
        fontsize=12,
        fontweight="bold",
        color="#1f1f1f",
        va="bottom",
    )
    settings_row_list = [
        ["Target model", target_model_name],
        ["Coverage threshold", f"{coverage_threshold:.2f}"],
        ["Disagreement inputs", f"{number_of_disagreement_inducing_inputs_found}"],
        ["Neuron coverage", f"{final_average_neuron_coverage:.4f}"],
    ]
    settings_table = settings_axis.table(
        cellText=settings_row_list,
        colLabels=["Setting", "Value"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.02, 0.58, 0.88],
        colWidths=[0.36, 0.22],
    )
    settings_table.auto_set_font_size(False)
    settings_table.set_fontsize(11)

    for (row_index, column_index), cell in settings_table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.8)
        cell.PAD = 0.12
        if row_index == 0:
            cell.set_facecolor("#f6f6f6")
            cell.set_text_props(weight="bold", color="#1f1f1f")
        else:
            cell.set_facecolor("white")
            if column_index == 0:
                cell.set_text_props(color="#666666")
            else:
                cell.set_text_props(color="#1f1f1f")

    table_axis = figure.add_subplot(grid_spec[2])
    table_axis.axis("off")
    table_axis.text(
        0.0,
        1.03,
        "Model Coverage Change",
        fontsize=13,
        fontweight="bold",
        color="#1f1f1f",
        va="bottom",
    )

    table_row_list = []
    for model_name, coverage_summary in coverage_gain_summary_dict.items():
        table_row_list.append(
            [
                model_name,
                f"{coverage_summary['baseline_neuron_coverage_ratio']:.4f}",
                f"{coverage_summary['updated_neuron_coverage_ratio']:.4f}",
                f"+{coverage_summary['added_neuron_count']}",
            ]
        )

    summary_table = table_axis.table(
        cellText=table_row_list,
        colLabels=["Model", "Baseline", "Updated", "Delta"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.10, 1.0, 0.82],
        colWidths=[0.46, 0.18, 0.18, 0.18],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(10.8)

    for (row_index, column_index), cell in summary_table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.8)
        cell.PAD = 0.10
        cell.get_text().set_wrap(True)
        cell.get_text().set_clip_on(True)
        if column_index == 0:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_ha("center")
        if row_index == 0:
            cell.set_facecolor("#f6f6f6")
            cell.set_text_props(weight="bold", color="#1f1f1f")
        else:
            cell.set_facecolor("white")
            if column_index == 3:
                cell.set_text_props(weight="bold", color="#1f1f1f")
            else:
                cell.set_text_props(color="#333333")

    if number_of_disagreement_inducing_inputs_found == 0:
        footer_text = "No new disagreement-inducing inputs found."
    else:
        footer_text = "Coverage deltas are measured after adding generated disagreement-inducing inputs."
    figure.text(
        0.06,
        0.05,
        footer_text,
        fontsize=10.5,
        color="#666666",
        va="bottom",
    )

    figure.savefig(output_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

def save_experiment_report(
    output_path,
    transformation,
    target_model_name,
    coverage_threshold,
    model_summary_list,
    clean_disagreement_count,
    clean_disagreement_rate,
    number_of_disagreement_inducing_inputs_found,
    final_average_neuron_coverage,
    coverage_gain_summary_dict,
    generated_input_summary_list,
):
    report_line_list = [
        "# Experiment Report",
        "",
        "## Headline",
        f"- Transformation: `{transformation}`",
        f"- Target model: `{target_model_name}`",
        f"- Coverage threshold: `{coverage_threshold:.2f}`",
        f"- Number of disagreement inducing inputs found: `{number_of_disagreement_inducing_inputs_found}`",
        f"- Neuron coverage achieved: `{final_average_neuron_coverage:.4f}`",
        f"- Clean disagreement count: `{clean_disagreement_count}`",
        f"- Clean disagreement rate: `{clean_disagreement_rate:.4f}`",
        "",
        "## Model Performance",
        "| Model | Saved Epoch | Val Acc | Test Loss | Test Acc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for model_summary in model_summary_list:
        report_line_list.append(
            "| "
            f"{model_summary['model_name']} | "
            f"{model_summary['saved_epoch']} | "
            f"{model_summary['validation_accuracy']:.4f} | "
            f"{model_summary['test_loss']:.4f} | "
            f"{model_summary['test_accuracy']:.4f} |"
        )

    report_line_list.extend(
        [
            "",
            "## Coverage Change",
            "| Model | Baseline | Generated | Added Neurons |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for model_name, coverage_summary in coverage_gain_summary_dict.items():
        report_line_list.append(
            "| "
            f"{model_name} | "
            f"{coverage_summary['baseline_neuron_coverage_ratio']:.4f} "
            f"({coverage_summary['baseline_covered_neuron_count']}/{coverage_summary['total_neuron_count']}) | "
            f"{coverage_summary['updated_neuron_coverage_ratio']:.4f} "
            f"({coverage_summary['updated_covered_neuron_count']}/{coverage_summary['total_neuron_count']}) | "
            f"+{coverage_summary['added_neuron_count']} |"
        )

    report_line_list.extend(
        [
            "",
            "## Generated Suspicious Inputs",
            "| Seed | True | Clean Prediction | Generated Prediction | Iters |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )

    for generated_input_summary in generated_input_summary_list:
        report_line_list.append(
            "| "
            f"{generated_input_summary['seed_index']} | "
            f"{generated_input_summary['true_label']} | "
            f"({generated_input_summary['original_prediction_wo_aug']}, "
            f"{generated_input_summary['original_prediction_w_aug']}) | "
            f"({generated_input_summary['generated_prediction_wo_aug']}, "
            f"{generated_input_summary['generated_prediction_w_aug']}) | "
            f"{generated_input_summary['iteration_count']} |"
        )

    if len(generated_input_summary_list) == 0:
        report_line_list.append("| - | - | - | - | - |")

    report_line_list.extend(
        [
            "",
            "## Saved Files",
            "- `coverage_summary.png`",
            "- `clean_disagreements/`",
            "- `generated_inputs/`",
            "",
        ]
    )

    output_path.write_text("\n".join(report_line_list))
