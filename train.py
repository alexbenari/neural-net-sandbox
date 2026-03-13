import argparse
import collections
import math
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


def compute_lr_scale(elapsed, time_budget, warmup_fraction, min_lr_ratio):
    if time_budget <= 0:
        return 1.0
    progress = min(max(elapsed / time_budget, 0.0), 1.0)
    if warmup_fraction > 0.0 and progress < warmup_fraction:
        return progress / warmup_fraction
    if progress >= 1.0:
        return min_lr_ratio
    decay_progress = (progress - warmup_fraction) / max(1e-8, 1.0 - warmup_fraction)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

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
    lr_schedule="constant",
    warmup_fraction=0.03,
    min_lr_ratio=0.1,
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
    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", learning_rate)

    training_set = prep.maybe_preload_dataset(training_set, device, preload)
    data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    dataset_size = len(training_set)
    print(f"Dataset size: {dataset_size}, batch size: {batch_size}, epochs: {epochs}")
    
    total_training_time = 0
    train_start = time.perf_counter()
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
            if lr_schedule == "cosine":
                elapsed = time.perf_counter() - train_start
                lr_scale = compute_lr_scale(
                    elapsed,
                    TIME_BUDGET,
                    warmup_fraction=warmup_fraction,
                    min_lr_ratio=min_lr_ratio,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["initial_lr"] * lr_scale
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
        if input_type == "digit1h":
            model = model_map[input_type](
                digit_embedding_dim=args.digit_embed_dim,
                tower_hidden_dim=args.tower_hidden_dim,
                tower_hidden_layers=args.tower_hidden_layers,
                chunk_dim=args.chunk_dim,
                attn_heads=args.attn_heads,
                ff_hidden_dim=args.ff_hidden_dim,
                head_hidden_dim=args.head_hidden_dim,
                head_hidden_layers=args.head_hidden_layers,
                activation_name=args.activation,
                pooling=args.pooling,
                use_digit_position_embedding=not args.no_digit_pos_embedding,
                use_chunk_position_embedding=args.chunk_pos_embedding,
                use_attention=not args.no_attention,
                use_ff=not args.no_ff,
            )
        else:
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
            lr_schedule=args.lr_schedule,
            warmup_fraction=args.warmup_fraction,
            min_lr_ratio=args.min_lr_ratio,
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
        default="adamw",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0004,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="cosine",
        help="Learning-rate schedule to use during training.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.03,
        help="Fraction of the time budget reserved for linear LR warmup.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.02,
        help="Final LR as a fraction of the base LR when using cosine decay.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay for optimizers that support it (e.g., adamw).",
    )
    parser.add_argument(
        "--digit-embed-dim",
        type=int,
        default=24,
        help="Digit embedding width for the digit1h tower model.",
    )
    parser.add_argument(
        "--tower-hidden-dim",
        type=int,
        default=256,
        help="Hidden width for chunk tower MLP layers.",
    )
    parser.add_argument(
        "--tower-hidden-layers",
        type=int,
        default=3,
        help="Number of hidden layers in the chunk tower MLP.",
    )
    parser.add_argument(
        "--chunk-dim",
        type=int,
        default=64,
        help="Chunk state width used by attention and the output head.",
    )
    parser.add_argument(
        "--attn-heads",
        type=int,
        default=2,
        help="Attention head count for the digit1h tower model.",
    )
    parser.add_argument(
        "--ff-hidden-dim",
        type=int,
        default=128,
        help="Hidden width for the post-attention feedforward block.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=128,
        help="Hidden width for the final prediction head.",
    )
    parser.add_argument(
        "--head-hidden-layers",
        type=int,
        default=1,
        help="Number of hidden layers in the final prediction head.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu", "silu"],
        default="silu",
        help="Activation to use inside the digit1h tower model.",
    )
    parser.add_argument(
        "--pooling",
        choices=["flatten", "mean"],
        default="flatten",
        help="How to pool chunk states before the final head.",
    )
    parser.add_argument(
        "--no-digit-pos-embedding",
        action="store_true",
        help="Disable learned digit position embeddings in the digit1h tower model.",
    )
    parser.add_argument(
        "--chunk-pos-embedding",
        action="store_true",
        default=False,
        help="Enable learned chunk position embeddings in the digit1h tower model.",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable cross-chunk self-attention in the digit1h tower model.",
    )
    parser.add_argument(
        "--no-ff",
        action="store_true",
        help="Disable the post-attention feedforward residual block in the digit1h tower model.",
    )
    
    return parser


if __name__ == "__main__":
    main()
