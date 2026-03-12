import argparse
import collections
from html import parser
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import prepare as prep
from prepare import TIME_BUDGET


from nets import (
    SimpleMLPForInt,
    SimpleMLPForDigits,
    SimpleMLPForBinary,
    TowersMLPForDigit1H,
)

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
    global_step=0,
):
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params}")

    criterion = prep.make_criterion()

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

    training_set = prep.maybe_preload_dataset(training_set, device, preload)
    data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dataset_size = len(training_set)
    print(f"Dataset size: {dataset_size}, batch size: {batch_size}, epochs: {epochs}")
    
    total_training_time = 0
    for epoch in range(epochs):
        if(total_training_time >= TIME_BUDGET):
            print(f"Time budget of {TIME_BUDGET} seconds reached. Stopping training.")
            break
        epoch_start = time.perf_counter()
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        seen = 0
        for _, (data, labels) in enumerate(data_loader, start=1):
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
            global_step += 1
            batch_size_actual = labels.size(0)
            running_loss += loss.detach() * batch_size_actual
            correct += (rounded_outputs == labels).sum()
            seen += batch_size_actual
            
        avg_loss = (running_loss / max(seen, 1)).item()
        accuracy = (correct / max(seen, 1)).item()
        elapsed = time.perf_counter() - epoch_start
        total_training_time += elapsed
        if log_per_epoch:
            print(
                f"Epoch [{epoch + 1}/{epochs}] avg_loss: {avg_loss:.4f} "
                f"accuracy: {accuracy:.4f}"
                f"epoch time: {elapsed:.2f}s"
                f"remaining time: {TIME_BUDGET - total_training_time:.2f}s"
            )
    
    print(
        f"Train avg_loss: {avg_loss:.4f} accuracy: {accuracy:.4f} "
    )    
    return avg_loss, accuracy


def wrap_dataset_for_input_type(dataset, input_type):
    if input_type == "normalized-int":
        dataset = prep.NormalizedIntWrapper(dataset)
    if input_type in ("digit", "digit1h"):
        dataset = prep.DigitsDatasetWrapper(dataset)
    if input_type == "digit1h":
        dataset = prep.DigitOneHotWrapper(dataset)
    if input_type == "binary":
        dataset = prep.BinaryDatasetWrapper(dataset)
    return dataset


def main():
    parser = cmdline_parser()
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    model_map = {
        "normalized-int": SimpleMLPForInt,
        "digit": SimpleMLPForDigits,
        #"digit1h": SimpleMLPForDigit1H,
        "digit1h": TowersMLPForDigit1H,
        "binary": SimpleMLPForBinary,
    }
    
    if args.input_type and not args.train:
        parser.error("--input-type requires --train.")

    if args.weight_decay is not None:
        supported = {"sgd", "adam", "adamw", "rmsprop", "adagrad"}
        if args.optimizer not in supported:
            parser.error(
                f"--weight-decay is not supported for optimizer {args.optimizer}."
            )

    if args.train:
        input_type = args.input_type or "normalized-int"
        dataset_map = {
            "normalized-int": prep.IntDataset,
            "digit": prep.IntDataset,
            "digit1h": prep.IntDataset,
            "binary": prep.IntDataset,
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

        train_loss, train_accuracy = train(
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
        )
        run_eval = True
        run_test = True
        if run_eval:
            eval_dataset = dataset_cls(os.path.join("data", f"{prefix}-eval.csv"))
            eval_dataset = wrap_dataset_for_input_type(eval_dataset, input_type)
            eval_loss, eval_accuracy = prep.evaluate(
                model,
                eval_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Eval",
            )
        if run_test:
            test_loss, test_accuracy = prep.evaluate(
                model,
                test_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Test",
            )
        return

    parser.print_help()

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
        "--train",
        action="store_true",
        help="Run the training loop.",
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
        default=500,
        help="Number of training epochs when running training.",
    )
    
    parser.add_argument(
        "--input-type",
        choices=["normalized-int", "digit", "digit1h", "binary"],
        default="digit1h",
        help=(
            "Input format to use for training: normalized-int, digit, digit1h, "
            "or binary."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=True,
        help="Preload the dataset to the training device before training.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "rmsprop", "adagrad"],
        default="adam",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0004,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for optimizers that support it (e.g., adamw).",
    )
    
    return parser


if __name__ == "__main__":
    main()
