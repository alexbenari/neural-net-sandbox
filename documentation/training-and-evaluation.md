# Training and Evaluation

This project trains a small MLP on integer inputs and uses wrappers to
represent the same data as digits, one-hot digits, or binary inputs.

## Quick start

```bash
python sandbox.py --train --input-type normalized-int
```

## Common training scenarios

Train with a specific input representation:

```bash
python sandbox.py --train --input-type digit
python sandbox.py --train --input-type digit1h
python sandbox.py --train --input-type binary
```

Customize optimizer and learning rate:

```bash
python sandbox.py --train --input-type digit1h --optimizer adam --lr 0.0003
```

Control batch size and epochs:

```bash
python sandbox.py --train --input-type normalized-int --batch-size 128 --epochs 200
```

Run sanity checks (short, targeted checks):

```bash
python sandbox.py --train --input-type digit1h --sanity --sanity-epochs 10 --sanity-dataset-size 1000
```

Generate datasets (int-only; other formats are derived via wrappers):

```bash
python sandbox.py --generate-data
```

## Saving models

Save a trained model (with metadata) under `models/`:

```bash
python sandbox.py --train --input-type digit1h --save-model
```

Save with a custom name:

```bash
python sandbox.py --train --input-type digit1h --save-model my_digit1h_run
```

List saved models:

```bash
python sandbox.py --models
```

## Evaluating saved models

Evaluate a saved model by name (runs both test and eval datasets):

```bash
python sandbox.py --eval my_digit1h_run
```

## TensorBoard

Log to TensorBoard during training (automatic). View logs:

```bash
tensorboard --logdir runs
```

For a specific run:

```bash
tensorboard --logdir runs/digit1h-20260118-104532
```
