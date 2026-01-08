import argparse
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_generation import DataSetGenerator, IntDataset, DigitsDataset, BinaryDataset
from nets import SimpleMLPForInt, SimpleMLPForDigits, SimpleMLPForBinary
from number_words import number_to_words

def make_criterion():
    return lambda output, target: torch.clamp(
        torch.abs(output - target) - 0.5,
        min=0.0,
    ).mean()


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


def train(
    model,
    training_set,
    epochs=500,
    batch_size=32,
    optimizer_name="sgd",
    learning_rate=0.01,
    device="cpu",
    preload=True,
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
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        seen = 0
        for data, labels in data_loader:
            if not preload:
                data = data.to(device)
                labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size_actual = labels.size(0)
            running_loss += loss.detach() * batch_size_actual
            correct += (torch.round(outputs) == labels).sum()
            seen += batch_size_actual
        avg_loss = (running_loss / max(seen, 1)).item()
        accuracy = (correct / max(seen, 1)).item()
        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch [{epoch + 1}/{epochs}] avg loss: {avg_loss:.4f} "
            f"accuracy: {accuracy:.4f} time: {elapsed:.2f}s"
        )


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
            correct += (torch.round(outputs) == labels).sum()
            seen += batch_size_actual
    avg_loss = (running_loss / max(seen, 1)).item()
    accuracy = (correct / max(seen, 1)).item()
    print(f"{label} avg loss: {avg_loss:.4f} accuracy: {accuracy:.4f}")
    model.train()


def main():
    parser = cmdline_parser()
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    device = "cuda" if cuda_available else "cpu"

    if args.input_type and not args.train:
        parser.error("--input-type requires --train.")
    if (args.test or args.eval) and not args.train:
        parser.error("--test and --eval require --train.")

    if args.train:
        input_type = args.input_type or "int"
        dataset_map = {
            "int": IntDataset,
            "digit": DigitsDataset,
            "binary": BinaryDataset,
        }
        prefix_map = {
            "int": "int",
            "digit": "digits",
            "binary": "binary",
        }

        model_map = {
            "int": SimpleMLPForInt,
            "digit": SimpleMLPForDigits,
            "binary": SimpleMLPForBinary,
        }

        dataset_cls = dataset_map[input_type]
        prefix = prefix_map[input_type]
        train_dataset = dataset_cls(os.path.join("data", f"{prefix}-train.csv"))
        test_dataset = dataset_cls(os.path.join("data", f"{prefix}-test.csv"))
        model = model_map[input_type]()
        train(
            model,
            train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer_name=args.optimizer,
            learning_rate=args.lr,
            device=device,
            preload=args.preload,
        )
        run_test = args.test or (not args.test and not args.eval)
        run_eval = args.eval or (not args.test and not args.eval)
        if run_test:
            evaluate(
                model,
                test_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Test",
            )
        if run_eval:
            eval_dataset = dataset_cls(os.path.join("data", f"{prefix}-eval.csv"))
            evaluate(
                model,
                eval_dataset,
                batch_size=args.batch_size,
                device=device,
                preload=args.preload,
                label="Eval",
            )
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
        "  --input-type: int | digit | binary\n"
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
        "--input-type",
        choices=["int", "digit", "binary"],
        default=None,
        help="Input format to use for training: int, digit, or binary.",
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
        "--test",
        action="store_true",
        help="Run evaluation on the test set after training.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation on the eval set after training.",
    )
    
    return parser


if __name__ == "__main__":
    main()
