import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_generation import (
    IntDataset,
    NormalizedIntWrapper,
    DigitsDatasetWrapper,
    DigitOneHotWrapper,
    BinaryDatasetWrapper,
)

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)

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
    print(
        f"{label} avg_loss: {avg_loss:.4f} accuracy: {accuracy:.4f} "
    )
    return avg_loss, accuracy


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


def get_evaluation_sets(input_type):
    dataset_cls = IntDataset
    test_dataset = dataset_cls(os.path.join("data", "int-test.csv"))
    eval_dataset = dataset_cls(os.path.join("data", "int-eval.csv"))
    test_dataset = wrap_dataset_for_input_type(test_dataset, input_type)
    eval_dataset = wrap_dataset_for_input_type(eval_dataset, input_type)
    return test_dataset,eval_dataset


