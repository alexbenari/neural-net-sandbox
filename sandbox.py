import argparse
import collections
import copy
import json
import os
import random
import time
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter

from data_generation import (
    DataSetGenerator,
    IntDataset,
    NormalizedIntWrapper,
    DigitsDatasetWrapper,
    DigitOneHotWrapper,
    BinaryDatasetWrapper,
    ZeroedInputsDataset,
)
from nets import (
    SimpleMLPForInt,
    SimpleMLPForDigits,
    SimpleMLPForDigit1H,
    SimpleMLPForBinary,
    TowersMLPForDigit1H,
    make_mlp,
)
from number_words import number_to_words

def make_criterion():
    return lambda output, target: (torch.abs(output - target)).mean()


def maybe_preload_dataset(dataset, device, preload):
    if not preload:
        return dataset
    data_list = []
    label_list = []
    for data, label in dataset:
        data_list.append(data)
        label_list.append(label)
    data_tensor = torch.stack(data_list).to(device)
    label_tensor = torch.stack(label_list).to(device)
    return TensorDataset(data_tensor, label_tensor)

def _ints_from_inputs(data, input_type):
    if input_type == "digit":
        values = []
        for row in data:
            digits = "".join(str(int(value.item())) for value in row)
            values.append(int(digits))
        return values
    if input_type == "digit1h":
        values = []
        for row in data:
            digits = row.view(-1, 10).argmax(dim=1).tolist()
            values.append(int("".join(str(d) for d in digits)))
        return values
    if input_type == "binary":
        values = []
        for row in data:
            bits = "".join(str(int(value.item())) for value in row)
            values.append(int(bits, 2))
        return values
    if input_type == "normalized-int":
        return [int(round(value.item() * 1_000_000_000)) for value in data.view(-1)]
    return [int(round(value.item())) for value in data.view(-1)]


def train(
    model,
    training_set,
    epochs=500,
    batch_size=32,
    optimizer_name="sgd",
    learning_rate=0.01,
    weight_decay=None,
    device="cpu",
    preload=True,
    log_per_epoch=True,
    input_type="normalized-int",
    verbose_samples=False,
    writer=None,
    global_step=0,
):
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params}")

    criterion = make_criterion()

    optimizer_map = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
    }
    if weight_decay is not None:
        optimizer = optimizer_map[optimizer_name](
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = optimizer_map[optimizer_name](model.parameters(), lr=learning_rate)

    training_set = maybe_preload_dataset(training_set, device, preload)
    data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dataset_size = len(training_set)
    print(f"Dataset size: {dataset_size}, batch size: {batch_size}, epochs: {epochs}")
    # Skip add_graph to avoid sync overhead in training.
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        within_half = torch.tensor(0.0, device=device)
        within_one = torch.tensor(0.0, device=device)
        seen = 0
        last_samples = collections.deque(maxlen=20)
        for batch_idx, (data, labels) in enumerate(data_loader, start=1):
            if not preload:
                data = data.to(device)
                labels = labels.to(device)
            model_inputs = None
            inputs = None
            if verbose_samples:
                data_cpu = data.detach().cpu()
                inputs = data_cpu.view(data_cpu.size(0), -1)
                model_inputs = []
                for row in inputs:
                    row_list = row.tolist()
                    if input_type == "normalized-int":
                        model_inputs.append(f"{row_list[0]:.9f}")
                    else:
                        model_inputs.append(str(row_list))
            outputs = model(data)
            assert outputs.shape == labels.shape
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rounded_outputs = torch.round(outputs)
            abs_error = torch.abs(outputs - labels)
            global_step += 1
            batch_size_actual = labels.size(0)
            running_loss += loss.detach() * batch_size_actual
            correct += (rounded_outputs == labels).sum()
            within_half += (abs_error < 0.5).sum()
            within_one += (abs_error < 1.0).sum()
            seen += batch_size_actual
            if verbose_samples:
                targets = labels.detach().cpu().view(-1).tolist()
                preds = outputs.detach().cpu().view(-1).tolist()
                print(f"Batch {batch_idx} samples:")
                for idx, (input_value, target, pred) in enumerate(
                    zip(inputs, targets, preds)
                ):
                    model_input_repr = model_inputs[idx]
                    rounded = round(pred)
                    abs_err = abs(pred - target)
                    line = (
                        f"sample input={input_value} model_input={model_input_repr} "
                        f"target={int(target)} "
                        f"pred={pred:.4f} rounded={rounded} abs_err={abs_err:.4f}"
                    )
                    print(line)
                    last_samples.append(line)
        avg_loss = (running_loss / max(seen, 1)).item()
        accuracy = (correct / max(seen, 1)).item()
        within_half_ratio = (within_half / max(seen, 1)).item()
        within_one_ratio = (within_one / max(seen, 1)).item()
        elapsed = time.perf_counter() - epoch_start
        if writer is not None:
            writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            writer.add_scalar("train/epoch_accuracy", accuracy, epoch + 1)
            writer.add_scalar("train/epoch_within_0p5", within_half_ratio, epoch + 1)
            writer.add_scalar("train/epoch_within_1p0", within_one_ratio, epoch + 1)
        if log_per_epoch:
            print(
                f"Epoch [{epoch + 1}/{epochs}] avg loss: {avg_loss:.4f} "
                f"accuracy: {accuracy:.4f} within0.5: {within_half_ratio:.4f} "
                f"within1.0: {within_one_ratio:.4f} time: {elapsed:.2f}s"
            )
        if verbose_samples and last_samples:
            print(f"Epoch [{epoch + 1}/{epochs}] last 20 samples:")
            for line in last_samples:
                print(line)
    return avg_loss, accuracy, global_step


def evaluate(model, dataset, batch_size=32, device="cpu", preload=True, label="Test"):
    criterion = make_criterion()
    dataset = maybe_preload_dataset(dataset, device, preload)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    within_half = torch.tensor(0.0, device=device)
    within_one = torch.tensor(0.0, device=device)
    seen = 0
    with torch.no_grad():
        for data, labels in data_loader:
            if not preload:
                data = data.to(device)
                labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            batch_size_actual = labels.size(0)
            running_loss += loss.detach() * batch_size_actual
            rounded_outputs = torch.round(outputs)
            correct += (rounded_outputs == labels).sum()
            abs_error = torch.abs(outputs - labels)
            within_half += (abs_error < 0.5).sum()
            within_one += (abs_error < 1.0).sum()
            seen += batch_size_actual
    avg_loss = (running_loss / max(seen, 1)).item()
    accuracy = (correct / max(seen, 1)).item()
    within_half_ratio = (within_half / max(seen, 1)).item()
    within_one_ratio = (within_one / max(seen, 1)).item()
    print(
        f"{label} avg loss: {avg_loss:.4f} accuracy: {accuracy:.4f} "
        f"within0.5: {within_half_ratio:.4f} within1.0: {within_one_ratio:.4f}"
    )
    return avg_loss, accuracy, within_half_ratio, within_one_ratio


def loss_vs_baseline_loss(model, dataset, device, batch_size, preload):
    model = model.to(device)
    model.eval()
    criterion = make_criterion()
    dataset = maybe_preload_dataset(dataset, device, preload)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data, labels = next(iter(loader))
    if not preload:
        data = data.to(device)
        labels = labels.to(device)
    with torch.no_grad():
        outputs = model(data)
        loss = criterion(outputs, labels)
    prediction_stats_for_batch(outputs)
    
    median_label_value = labels.median().item()
    baseline_loss = criterion(torch.full_like(labels, median_label_value), labels).item()
    batch_items = labels.size(0)
    avg_loss = (loss.item() * batch_items) / max(batch_items, 1)
    ratio = avg_loss / baseline_loss if baseline_loss > 0 else float('inf')
    print(f"Batch label median value: {median_label_value:.8f}")
    print(f"Baseline average loss: {baseline_loss:.8f}")
    print(f"Avg loss: {avg_loss:.8f}")
    print(f"Loss ratio (avg/baseline): {ratio:.8f}")
    if not (0.75 <= ratio <= 1.25):
        print("ERROR: Loss vs baseline sanity check failed.")
    else: print("Loss vs baseline sanity check succeeded.")


def train_vs_zeroed_inputs(
    model,
    dataset,
    eval_dataset,
    device,
    batch_size,
    optimizer_name,
    learning_rate,
    weight_decay,
    preload,
    epochs,
    input_type,
):
    print(
        "Sanity check: training with real inputs vs zeroed inputs "
        f"({epochs} epochs, input_type={input_type})."
    )
    real_model = copy.deepcopy(model)
    zeroed_model = copy.deepcopy(model)

    train(
        real_model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        preload=preload,
        input_type=input_type,
        log_per_epoch=False,
        writer=None,
    )
    zeroed_dataset = ZeroedInputsDataset(dataset)
    train(
        zeroed_model,
        zeroed_dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        preload=preload,
        input_type=input_type,
        log_per_epoch=False,
        writer=None,
    )

    _, real_accuracy, _, _ = evaluate(
        real_model,
        eval_dataset,
        batch_size=batch_size,
        device=device,
        preload=True,
        label="Sanity/Real inputs",
    )
    _, zeroed_accuracy, _, _ = evaluate(
        zeroed_model,
        eval_dataset,
        batch_size=batch_size,
        device=device,
        preload=True,
        label="Sanity/Zeroed inputs",
    )

    diff = real_accuracy - zeroed_accuracy
    if diff <= 0:
        print("ERROR: Zeroed-inputs accuracy matches or beats real-inputs accuracy.")
    else:
        print(f"Real inputs beat zeroed inputs by {diff:.4f} accuracy.")


def prediction_stats_for_batch(outputs):
    preds = outputs.view(-1)
    pred_mean = preds.mean().item()
    pred_median = preds.median().item()
    pred_min = preds.min().item()
    pred_max = preds.max().item()
    print("Batch size:", preds.size(0))
    print(
        "Prediction stats for batch (mean/median/min/max): "
        f"{pred_mean:.8f}/{pred_median:.8f}/{pred_min:.8f}/{pred_max:.8f}"
    )


def model_sanity(
    model,
    dataset,
    eval_dataset,
    device,
    batch_size,
    optimizer_name,
    learning_rate,
    weight_decay,
    preload,
    epochs,
    input_type,
):
    overfit_to_small_batch(
        model,
        dataset,
        device,
        batch_size,
        optimizer_name,
        learning_rate,
        preload,
        epochs,
        input_type,
    )
    loss_vs_baseline_loss(model, dataset, device, batch_size, preload)
    train_vs_zeroed_inputs(
        model,
        dataset,
        eval_dataset,
        device,
        batch_size,
        optimizer_name,
        learning_rate,
        weight_decay,
        preload,
        epochs,
        input_type,
    )


def wrap_dataset_for_input_type(dataset, input_type):
    if input_type == "normalized-int":
        dataset = NormalizedIntWrapper(dataset)
    if input_type in ("digit", "digit1h"):
        dataset = DigitsDatasetWrapper(dataset)
    if input_type == "digit1h":
        dataset = DigitOneHotWrapper(dataset)
    if input_type == "binary":
        dataset = BinaryDatasetWrapper(dataset)
    return dataset


def _model_name_for_save(
    input_type,
    optimizer_name,
    learning_rate,
    batch_size,
    epochs,
    weight_decay,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    lr_value = str(learning_rate).replace(".", "p")
    if weight_decay is not None:
        wd_value = str(weight_decay).replace(".", "p")
        wd_label = f"-wd{wd_value}"
    else:
        wd_label = ""
    return (
        f"{input_type}-{optimizer_name}-lr{lr_value}{wd_label}-bs{batch_size}-"
        f"ep{epochs}-{timestamp}"
    )


def _sanitize_model_name(model_name):
    if model_name is None:
        return None
    sanitized = model_name
    for ch in (os.path.sep, os.path.altsep, ":", "/", "\\"):
        if ch:
            sanitized = sanitized.replace(ch, "_")
    return sanitized


def _model_path_from_name(model_name):
    model_name = _sanitize_model_name(model_name)
    filename = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    return os.path.join("models", filename)


def save_model(model, metadata, model_name):
    os.makedirs("models", exist_ok=True)
    model_name = _sanitize_model_name(model_name)
    path = _model_path_from_name(model_name)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)
    metadata_path = os.path.join("models", f"{model_name}.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Saved model to {path}")


def list_models():
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return []
    entries = [
        entry
        for entry in os.scandir(models_dir)
        if entry.is_file() and entry.name.endswith(".pt")
    ]
    entries.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    return [os.path.splitext(entry.name)[0] for entry in entries]


def evaluate_and_log(writer, model, dataset, batch_size, device, preload, label, tag):
    loss, accuracy, within_half, within_one = evaluate(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        preload=preload,
        label=label,
    )
    writer.add_scalar(f"{tag}/final_loss", loss, 0)
    writer.add_scalar(f"{tag}/final_accuracy", accuracy, 0)
    writer.add_scalar(f"{tag}/final_within_0p5", within_half, 0)
    writer.add_scalar(f"{tag}/final_within_1p0", within_one, 0)
    return loss, accuracy


def load_model(model_path, model_map):
    state_dict, metadata = _load_checkpoint(model_path)
    input_type = metadata.get("input_type")
    if input_type == "int":
        input_type = "normalized-int"
    if input_type not in model_map:
        raise ValueError(f"Unknown input type in metadata: {input_type}")
    state_dict = _normalize_state_dict_keys(state_dict)
    model_sizes = metadata.get("model_sizes") or _infer_sizes_from_state_dict(
        state_dict
    )
    if model_sizes:
        model = make_mlp(model_sizes)
    else:
        model = model_map[input_type]()
    model.load_state_dict(state_dict)
    return model, metadata


def _infer_sizes_from_state_dict(state_dict):
    layer_items = []
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and key.startswith("layers."):
            parts = key.split(".")
            try:
                idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            out_features, in_features = tensor.shape
            layer_items.append((idx, in_features, out_features))
    if not layer_items:
        return []
    layer_items.sort()
    sizes = [layer_items[0][1]]
    sizes.extend(item[2] for item in layer_items)
    return sizes


def _load_checkpoint(model_path):
    metadata = _load_metadata_sidecar(model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    if not metadata:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            full_checkpoint = torch.load(model_path, map_location="cpu")
        metadata = full_checkpoint.get("metadata", {})
        if isinstance(full_checkpoint, dict) and "model_state" in full_checkpoint:
            state_dict = full_checkpoint["model_state"]
    return state_dict, metadata


def _load_metadata_sidecar(model_path):
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    metadata_path = os.path.join("models", f"{base_name}.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_state_dict_keys(state_dict):
    if any(key.startswith("layers.") for key in state_dict):
        return state_dict
    if all(key.startswith("net.layers.") for key in state_dict):
        normalized = {}
        for key, value in state_dict.items():
            normalized[key.replace("net.", "", 1)] = value
        return normalized
    return state_dict


def _extract_model_sizes(model):
    if not hasattr(model, "net"):
        return []
    layers = getattr(model.net, "layers", None)
    if not layers:
        return []
    sizes = [layers[0].in_features]
    sizes.extend(layer.out_features for layer in layers)
    return sizes


def print_model_parameters(model):
    total_params = 0
    print("Model parameters:")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        param_norm = param.detach().norm().item()
        print(
            f"- {name}: shape={tuple(param.shape)} params={num_params} "
            f"norm={param_norm:.6f}"
        )
    print(f"Total parameters: {total_params}")


def overfit_to_small_batch(
    model,
    dataset,
    device,
    batch_size,
    optimizer_name,
    learning_rate,
    weight_decay,
    preload,
    epochs,
    input_type,
):
    if len(dataset) < 2:
        raise ValueError("Overfit sanity check needs a dataset of at least 2 samples.")
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), 2)
    small_dataset = Subset(dataset, indices)
    print(
        "Sanity check: overfit a 2-sample batch "
        f"({epochs} epochs, input_type={input_type})."
    )
    overfit_model = copy.deepcopy(model)
    train(
        overfit_model,
        small_dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        preload=preload,
        input_type=input_type,
        log_per_epoch=False,
        writer=None,
    )
    avg_loss, _, _, _ = evaluate(
        overfit_model,
        small_dataset,
        batch_size=batch_size,
        device=device,
        preload=preload,
        label="Sanity/Overfit",
    )
    if avg_loss > 1e-6:
        print("ERROR: Overfit sanity check did not reach zero loss. The loss was:", avg_loss)
    #assert avg_loss <= 1e-6, "Overfit sanity check did not reach zero loss."


def _digits_tensor_to_int(digits):
    return int("".join(str(int(value.item())) for value in digits))


def _bits_tensor_to_int(bits):
    return int("".join(str(int(value.item())) for value in bits), 2)


def main():
    parser = cmdline_parser()
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    device = "cuda" if cuda_available else "cpu"

    if args.data_probe:
        from tests.data_probes import run_data_probes

        run_data_probes()
        return

    model_map = {
        "normalized-int": SimpleMLPForInt,
        "digit": SimpleMLPForDigits,
        #"digit1h": SimpleMLPForDigit1H,
        "digit1h": TowersMLPForDigit1H,
        "binary": SimpleMLPForBinary,
    }
    

    if args.models:
        models = list_models()
        if not models:
            print("No saved models found.")
        else:
            for name in models:
                print(name)
        return

    if args.model:
        model_path = _model_path_from_name(args.model)
        if not os.path.isfile(model_path):
            parser.error(f"Model not found: {model_path}")
        _, metadata = _load_checkpoint(model_path)
        if not metadata:
            print("No metadata found in model.")
        else:
            for key in sorted(metadata.keys()):
                print(f"{key}: {metadata[key]}")
        try:
            model, metadata = load_model(model_path, model_map)
        except Exception as exc:
            print(f"ERROR: Failed to load model: {exc}")
            return
        input_type = metadata.get("input_type")
        if input_type == "int":
            input_type = "normalized-int"
        print(f"Model input type: {input_type}")
        print_model_parameters(model)
        return

    if args.input_type and not args.train:
        parser.error("--input-type requires --train.")
    if args.sanity and not args.train:
        parser.error("--sanity requires --train.")
    if args.weight_decay is not None:
        supported = {"sgd", "adam", "adamw", "rmsprop", "adagrad"}
        if args.optimizer not in supported:
            parser.error(
                f"--weight-decay is not supported for optimizer {args.optimizer}."
            )

    if args.eval:
        model_path = _model_path_from_name(args.eval)
        if not os.path.isfile(model_path):
            parser.error(f"Model not found: {model_path}")
        model, metadata = load_model(model_path, model_map)
        input_type = metadata.get("input_type")
        if input_type not in model_map:
            parser.error(f"Unknown input type in metadata: {input_type}")
        test_dataset, eval_dataset = get_avaluation_sets(input_type)
        model = model.to(device)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"eval-{args.eval}-{timestamp}"
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
        batch_size = metadata.get("batch_size", args.batch_size)
        evaluate_and_log(
            writer,
            model,
            test_dataset,
            batch_size=batch_size,
            device=device,
            preload=args.preload,
            label="Test",
            tag="test",
        )
        evaluate_and_log(
            writer,
            model,
            eval_dataset,
            batch_size=batch_size,
            device=device,
            preload=args.preload,
            label="Eval",
            tag="eval",
        )
        writer.close()
        return

    if args.train:
        input_type = args.input_type or "normalized-int"
        dataset_map = {
            "normalized-int": IntDataset,
            "digit": IntDataset,
            "digit1h": IntDataset,
            "binary": IntDataset,
        }
        prefix_map = {
            "normalized-int": "int",
            "digit": "int",
            "digit1h": "int",
            "binary": "int",
        }

        dataset_cls = dataset_map[input_type]
        prefix = prefix_map[input_type]
        train_dataset = dataset_cls(os.path.join("data", f"{prefix}-train.csv"))
        test_dataset = dataset_cls(os.path.join("data", f"{prefix}-test.csv"))
        train_dataset = wrap_dataset_for_input_type(train_dataset, input_type)
        test_dataset = wrap_dataset_for_input_type(test_dataset, input_type)
        model = model_map[input_type]()
        if args.sanity:
            rng = random.Random(42)
            train_limit = min(args.sanity_dataset_size, len(train_dataset))
            if train_limit < len(train_dataset):
                train_indices = rng.sample(range(len(train_dataset)), train_limit)
                train_dataset = Subset(train_dataset, train_indices)

            eval_dataset = dataset_cls(os.path.join("data", f"{prefix}-eval.csv"))
            eval_limit = min(args.sanity_dataset_size, len(eval_dataset))
            if eval_limit < len(eval_dataset):
                eval_indices = rng.sample(range(len(eval_dataset)), eval_limit)
                eval_dataset = Subset(eval_dataset, eval_indices)

            train_dataset = wrap_dataset_for_input_type(train_dataset, input_type)
            eval_dataset = wrap_dataset_for_input_type(eval_dataset, input_type)
            model_sanity(
                model,
                train_dataset,
                eval_dataset,
                device,
                args.batch_size,
                args.optimizer,
                args.lr,
                args.weight_decay,
                args.preload,
                args.sanity_epochs,
                input_type,
            )
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name = None
        if args.save_model is not None and args.save_model != "":
            model_name = args.save_model
        run_name = args.run_name or model_name or f"{input_type}-{timestamp}"
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
        train_loss, train_accuracy, global_step = train(
            model,
            train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer_name=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            preload=args.preload,
            log_per_epoch=True,
            input_type=input_type,
            verbose_samples=args.verbose,
            writer=writer,
            global_step=0,
        )
        run_test = True
        run_eval = True
        final_loss = train_loss
        final_accuracy = train_accuracy
        final_source = "train"
        if run_test:
            test_loss, test_accuracy = evaluate_and_log(
                writer,
                model,
                test_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Test",
                tag="test",
            )
            final_loss = test_loss
            final_accuracy = test_accuracy
            final_source = "test"
        if run_eval:
            eval_dataset = dataset_cls(os.path.join("data", f"{prefix}-eval.csv"))
            eval_dataset = wrap_dataset_for_input_type(eval_dataset, input_type)
            eval_loss, eval_accuracy = evaluate_and_log(
                writer,
                model,
                eval_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Eval",
                tag="eval",
            )
            final_loss = eval_loss
            final_accuracy = eval_accuracy
            final_source = "eval"
        writer.close()
        if args.save_model is not None:
            model_name = model_name or _model_name_for_save(
                input_type,
                args.optimizer,
                args.lr,
                args.batch_size,
                args.epochs,
                args.weight_decay,
            )
            parameter_count = sum(p.numel() for p in model.parameters())
            metadata = {
                "input_type": input_type,
                "epochs_trained": args.epochs,
                "optimizer": args.optimizer,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
                "final_metrics_source": final_source,
                "model_sizes": _extract_model_sizes(model),
                "parameter_count": parameter_count,
            }
            save_model(model, metadata, model_name)
        return

    if args.generate_data:
        generator = DataSetGenerator(
            output_dir=args.output_dir,
            train_size=args.train_size,
            test_size=args.test_size,
            eval_size=args.eval_size,
        )
        generator.generate_all()
        return

    if args.spell is not None:
        value = args.spell
        words = number_to_words(value)
        digits_str = str(value).zfill(9)
        binary_str = format(value, "b").zfill(30)
        normalized_value = value / 1_000_000_000
        one_hot_chunks = []
        for ch in digits_str:
            chunk = ["0"] * 10
            chunk[int(ch)] = "1"
            one_hot_chunks.append("".join(chunk))
        print(f"Number: {value}")
        print(f"Words: {words}")
        print(f"Length: {len(words)}")
        print(f"normalized-int: {normalized_value:.9f}")
        print(f"digits: {digits_str}")
        print(f"digit1h: {' '.join(one_hot_chunks)}")
        print(f"binary: {binary_str}")
        return


    parser.print_help()

def get_avaluation_sets(input_type):
    dataset_cls = IntDataset
    test_dataset = dataset_cls(os.path.join("data", "int-test.csv"))
    eval_dataset = dataset_cls(os.path.join("data", "int-eval.csv"))
    test_dataset = wrap_dataset_for_input_type(test_dataset, input_type)
    eval_dataset = wrap_dataset_for_input_type(eval_dataset, input_type)
    return test_dataset,eval_dataset

def cmdline_parser():
    epilog = (
        "Legal options:\n"
        "  --input-type: normalized-int | digit | digit1h | binary\n"
        "  --optimizer: sgd | adam | adamw | rmsprop | adagrad\n"
    )
    parser = argparse.ArgumentParser(
        description="Neural net sandbox utilities.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate train/test/eval datasets under the data/ folder.",
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Run the training loop.",
    )
    group.add_argument(
        "--eval",
        type=str,
        help="Evaluate a saved model by name (from models/).",
    )
    group.add_argument(
        "--models",
        action="store_true",
        help="List available saved models.",
    )
    group.add_argument(
        "--model",
        type=str,
        help="Print metadata and parameters for a saved model by name (from models/).",
    )
    group.add_argument(
        "--spell",
        type=int,
        help="Print the spelled-out form of an integer and its length.",
    )
    group.add_argument(
        "--data-probe",
        action="store_true",
        help="Generate dataset probes and open the report.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for generated datasets.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100_000,
        help="Training set size for data generation.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10_000,
        help="Test set size for data generation.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=10_000,
        help="Total eval set size for data generation (includes 0..1000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs when running training.",
    )
    parser.add_argument(
        "--sanity-epochs",
        type=int,
        default=10,
        help="Number of epochs to run for sanity checks.",
    )
    parser.add_argument(
        "--sanity-dataset-size",
        type=int,
        default=1000,
        help="Subset size to use for sanity checks.",
    )
    parser.add_argument(
        "--input-type",
        choices=["normalized-int", "digit", "digit1h", "binary"],
        default=None,
        help=(
            "Input format to use for training: normalized-int, digit, digit1h, "
            "or binary."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=True,
        help="Preload the dataset to the training device before training.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_false",
        dest="preload",
        help="Disable dataset preloading.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "rmsprop", "adagrad"],
        default="sgd",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for optimizers that support it (e.g., adamw).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional TensorBoard run name (subfolder under runs/).",
    )
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="",
        default=None,
        help=(
            "Save the trained model under models/. "
            "Optionally provide a model name."
        ),
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run a single-sample loss check and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample predictions in training.",
    )
    
    return parser


if __name__ == "__main__":
    main()
