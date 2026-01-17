import argparse
import collections
import copy
import os
import random
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter

from data_generation import (
    DataSetGenerator,
    IntDataset,
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
    return [int(round(value.item())) for value in data.view(-1)]


def train(
    model,
    training_set,
    epochs=500,
    batch_size=32,
    optimizer_name="sgd",
    learning_rate=0.01,
    device="cpu",
    preload=True,
    log_per_epoch=True,
    input_type="int",
    verbose_samples=False,
    writer=None,
    global_step=0,
):
    model = model.to(device)
    print(model)

    criterion = make_criterion()

    optimizer_map = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
    }
    optimizer = optimizer_map[optimizer_name](model.parameters(), lr=learning_rate)

    training_set = maybe_preload_dataset(training_set, device, preload)
    data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dataset_size = len(training_set)
    print(f"Dataset size: {dataset_size}, batch size: {batch_size}, epochs: {epochs}")
    if writer is not None:
        sample_batch = next(iter(data_loader))[0]
        if not preload:
            sample_batch = sample_batch.to(device)
        writer.add_graph(model, sample_batch)
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        seen = 0
        last_samples = collections.deque(maxlen=20)
        for data, labels in data_loader:
            if not preload:
                data = data.to(device)
                labels = labels.to(device)
            outputs = model(data)
            assert outputs.shape == labels.shape
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rounded_outputs = torch.round(outputs)
            batch_accuracy = (rounded_outputs == labels).float().mean().item()
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/accuracy", batch_accuracy, global_step)
            global_step += 1
            batch_size_actual = labels.size(0)
            running_loss += loss.detach() * batch_size_actual
            correct += (rounded_outputs == labels).sum()
            seen += batch_size_actual
            if verbose_samples:
                inputs = _ints_from_inputs(data.detach().cpu(), input_type)
                targets = labels.detach().cpu().view(-1).tolist()
                preds = outputs.detach().cpu().view(-1).tolist()
                for input_value, target, pred in zip(inputs, targets, preds):
                    rounded = round(pred)
                    abs_err = abs(pred - target)
                    last_samples.append(
                        f"sample input={input_value} target={int(target)} "
                        f"pred={pred:.4f} rounded={rounded} abs_err={abs_err:.4f}"
                    )
        avg_loss = (running_loss / max(seen, 1)).item()
        accuracy = (correct / max(seen, 1)).item()
        elapsed = time.perf_counter() - epoch_start
        if writer is not None:
            writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            writer.add_scalar("train/epoch_accuracy", accuracy, epoch + 1)
        if log_per_epoch:
            print(
                f"Epoch [{epoch + 1}/{epochs}] avg loss: {avg_loss:.4f} "
                f"accuracy: {accuracy:.4f} time: {elapsed:.2f}s"
            )
        if verbose_samples and last_samples:
            print(f"Epoch [{epoch + 1}/{epochs}] last 20 samples:")
            for line in last_samples:
                print(line)
    return accuracy, global_step


def evaluate(model, dataset, batch_size=32, device="cpu", preload=True, label="Test"):
    criterion = make_criterion()
    dataset = maybe_preload_dataset(dataset, device, preload)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
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
            seen += batch_size_actual
    avg_loss = (running_loss / max(seen, 1)).item()
    accuracy = (correct / max(seen, 1)).item()
    print(f"{label} avg loss: {avg_loss:.4f} accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


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
        device=device,
        preload=preload,
        input_type=input_type,
        log_per_epoch=False,
        writer=None,
    )

    _, real_accuracy = evaluate(
        real_model,
        eval_dataset,
        batch_size=batch_size,
        device=device,
        preload=True,
        label="Sanity/Real inputs",
    )
    _, zeroed_accuracy = evaluate(
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
        preload,
        epochs,
        input_type,
    )


def overfit_to_small_batch(
    model,
    dataset,
    device,
    batch_size,
    optimizer_name,
    learning_rate,
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
        device=device,
        preload=preload,
        input_type=input_type,
        log_per_epoch=False,
        writer=None,
    )
    avg_loss, _ = evaluate(
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


def log_overfit_predictions(model, dataset, input_type, device, indices):
    model.eval()
    with torch.no_grad():
        for index in indices:
            data, target = dataset[index]
            if input_type == "digit1h":
                digits = data.view(-1, 10).argmax(dim=1)
                input_value = _digits_tensor_to_int(digits)
                model_input = data
            elif input_type == "digit":
                input_value = _digits_tensor_to_int(data)
                model_input = data
            elif input_type == "binary":
                input_value = _bits_tensor_to_int(data)
                model_input = data
            else:
                input_value = int(data.item())
                model_input = data

            model_input = model_input.to(device)
            target = target.to(device)
            output = model(model_input.unsqueeze(0))
            pred = output.squeeze(0)
            rounded = torch.round(pred)
            abs_error = torch.abs(pred - target.squeeze(0))
            print(
                "sample "
                f"input={input_value} "
                f"target={target.item():.0f} "
                f"pred={pred.item():.4f} "
                f"rounded={rounded.item():.0f} "
                f"abs_err={abs_error.item():.4f}"
            )
    model.train()


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

    if args.input_type and not args.train:
        parser.error("--input-type requires --train.")
    if (args.test or args.eval) and not args.train:
        parser.error("--test and --eval require --train.")
    if args.overfit and not args.train:
        parser.error("--overfit requires --train.")
    if args.sanity and not args.train:
        parser.error("--sanity requires --train.")

    if args.train:
        input_type = args.input_type or "int"
        dataset_map = {
            "int": IntDataset,
            "digit": IntDataset,
            "digit1h": IntDataset,
            "binary": IntDataset,
        }
        prefix_map = {
            "int": "int",
            "digit": "int",
            "digit1h": "int",
            "binary": "int",
        }

        model_map = {
            "int": SimpleMLPForInt,
            "digit": SimpleMLPForDigits,
            "digit1h": SimpleMLPForDigit1H,
            "binary": SimpleMLPForBinary,
        }

        dataset_cls = dataset_map[input_type]
        prefix = prefix_map[input_type]
        train_dataset = dataset_cls(os.path.join("data", f"{prefix}-train.csv"))
        test_dataset = dataset_cls(os.path.join("data", f"{prefix}-test.csv"))
        if input_type in ("digit", "digit1h"):
            train_dataset = DigitsDatasetWrapper(train_dataset)
            test_dataset = DigitsDatasetWrapper(test_dataset)
        if input_type == "digit1h":
            train_dataset = DigitOneHotWrapper(train_dataset)
            test_dataset = DigitOneHotWrapper(test_dataset)
        if input_type == "binary":
            train_dataset = BinaryDatasetWrapper(train_dataset)
            test_dataset = BinaryDatasetWrapper(test_dataset)
        if args.overfit:
            rng = random.Random(42)
            sample_size = min(args.overfit_size, len(train_dataset))
            indices = rng.sample(range(len(train_dataset)), sample_size)
            train_dataset = Subset(train_dataset, indices)
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

            if input_type in ("digit", "digit1h"):
                train_dataset = DigitsDatasetWrapper(train_dataset)
                eval_dataset = DigitsDatasetWrapper(eval_dataset)
            if input_type == "digit1h":
                train_dataset = DigitOneHotWrapper(train_dataset)
                eval_dataset = DigitOneHotWrapper(eval_dataset)
            if input_type == "binary":
                train_dataset = BinaryDatasetWrapper(train_dataset)
                eval_dataset = BinaryDatasetWrapper(eval_dataset)
            model_sanity(
                model,
                train_dataset,
                eval_dataset,
                device,
                args.batch_size,
                args.optimizer,
                args.lr,
                args.preload,
                args.sanity_epochs,
                input_type,
            )
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = args.run_name or f"{input_type}-{timestamp}"
        log_dir = os.path.join("runs", run_name)
        writer = SummaryWriter(log_dir=log_dir)
        accuracy, global_step = train(
            model,
            train_dataset,
            epochs=500 if args.overfit else args.epochs,
            batch_size=args.batch_size,
            optimizer_name=args.optimizer,
            learning_rate=args.lr,
            device=device,
            preload=args.preload,
            log_per_epoch=not args.overfit,
            input_type=input_type,
            verbose_samples=args.verbose,
            writer=writer,
            global_step=0,
        )
        if args.overfit:
            print(f"Overfit training accuracy: {accuracy:.4f}")
            writer.close()
            if args.verbose:
                base_dataset = dataset_cls(os.path.join("data", f"{prefix}-train.csv"))
                if input_type in ("digit", "digit1h"):
                    base_dataset = DigitsDatasetWrapper(base_dataset)
                if input_type == "digit1h":
                    base_dataset = DigitOneHotWrapper(base_dataset)
                if input_type == "binary":
                    base_dataset = BinaryDatasetWrapper(base_dataset)
                log_overfit_predictions(
                    model,
                    base_dataset,
                    input_type,
                    device,
                    indices,
                )
            return
        run_test = args.test or (not args.test and not args.eval)
        run_eval = args.eval or (not args.test and not args.eval)
        if run_test:
            test_loss, test_accuracy = evaluate(
                model,
                test_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Test",
            )
            writer.add_scalar("test/loss", test_loss, global_step)
            writer.add_scalar("test/accuracy", test_accuracy, global_step)
        if run_eval:
            eval_dataset = dataset_cls(os.path.join("data", f"{prefix}-eval.csv"))
            if input_type in ("digit", "digit1h"):
                eval_dataset = DigitsDatasetWrapper(eval_dataset)
            if input_type == "digit1h":
                eval_dataset = DigitOneHotWrapper(eval_dataset)
            if input_type == "binary":
                eval_dataset = BinaryDatasetWrapper(eval_dataset)
            eval_loss, eval_accuracy = evaluate(
                model,
                eval_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Eval",
            )
            writer.add_scalar("eval/loss", eval_loss, global_step)
            writer.add_scalar("eval/accuracy", eval_accuracy, global_step)
        writer.close()
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
        words = number_to_words(args.spell)
        print(words)
        print(len(words))
        return


    parser.print_help()

def cmdline_parser():
    epilog = (
        "Legal options:\n"
        "  --input-type: int | digit | digit1h | binary\n"
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
        choices=["int", "digit", "digit1h", "binary"],
        default=None,
        help="Input format to use for training: int, digit, digit1h, or binary.",
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
        "--run-name",
        default=None,
        help="Optional TensorBoard run name (subfolder under runs/).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on the test set after training.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation on the eval set after training.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Train on a small random subset and only report training accuracy.",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run a single-sample loss check and exit.",
    )
    parser.add_argument(
        "--overfit-size",
        type=int,
        default=256,
        help="Subset size to use for overfit mode.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample predictions in overfit mode.",
    )
    
    return parser


if __name__ == "__main__":
    main()
