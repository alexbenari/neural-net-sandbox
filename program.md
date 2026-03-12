# autoresearch

The LLM does its own research and continuously optimizes a model's loss.

## Terminology

**Research session**: A continuous flow of experiments
**Ground truth metric**: A single metric which defines the quality of the model. In our case this is average loss. The lower the loss, the better the model. Each experiment is an attempt to improve this metric.
**Experiment**: A single attempt to improve the ground truth metric. An experiment is based on an hypothesis, i.e. something that the agent thinks has a chance of improving the metric. It executes this hypothesis according to the steps detailed below.

## Trigger semantics

  If the user asks to "kick off", "start", "run", or "continue" a research session (or just "session"), that is authorization to enter the autonomous research loop immediately. It means: keep running back-to-back experiments until the user explicitly interrupts you. Do not stop to provide a summary after a successful or failed experiment. Do not stop for any reason whatsoever, just continue running experiments until you are stopped by the user.

## Continuous Execution 
NEVER STOP. Once the research session loop begins, your default behavior is continuous execution, not reporting. You may emit short progress updates, but those updates must not end the run. The only valid
  reasons to stop are:
  - the human explicitly tells you to stop, pause, or summarize
  - you hit a hard blocker that prevents any further experiment from running
  - the environment itself terminates the session
Absent one of those conditions, continue trying new ideas indefinitely.

## Setup

To set up a new research session:

1. **Choose a run tag**: pick a tag based on today's date (e.g. `mar5`) unless the user explicitly specifies one. The branch `neural-net-sandbox/<tag>` must not already exist — this is a fresh session. If the tag does exist, append a "-[num]" to the tag name, e.g. mar5-1. If that exists as well, find the next number which does not exist yet and use that as the appended [-num].
2. **Create the branch**: `git checkout -b neural-net-sandbox/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `data folder` contains the train, val and test sets. Do not modify.
   - `nets.py, train.py` — these are the files you modify. Model architecture, optimizer, training loop.

4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Initialize musings.md**: Create an empty `musings.md`. You will be writing to it before each idea you try
6. **Go**: Kick off the first experiment of the research session

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **limited time budget of 5 minutes at most**. You launch it simply as: `python.exe .\train.py --train`. The baseline (i.e. first) run uses the default cmd line param values. 

**What you CAN do:**
- Modify `train.py` and/or 'nets.py' — these are the only files you are allowed to edit. Within them, everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc. Many of these can be modified via command line args (e.g. learning rate, num epochs and others). Use existing command line args or add new ones or change the code directly (e.g. model structure). Try to restrict code changes to things like the structure of the MLP and similar ideas which do not naturally lend themselves to being expressed as simple cmd line args. 

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading and loss function definition code.
- Install new packages or add dependencies.
- Modify the evaluation harness. The avg_loss metric returned by the evaluate function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest avg_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always at most 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 avg_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 avg_loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline:
  python.exe .\train.py --train
After the baseline is logged, immediately continue into the experiment loop. The baseline run is not a stopping point.

## Output format

Once train.py finishes it prints a summary like this:

```
---
Train avg_loss: 8.3798 accuracy: 0.0440
Eval avg_loss: 7.7966 accuracy: 0.0393
Test avg_loss: 7.7757 accuracy: 0.0349
```
You can add logging of other bits of data as you see fit. 
You can extract the key metric from the log file:

```
findstr /R "^Eval avg_loss:" run.log
```

## Logging results

When an experiment run is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	avg_loss	status	description
```

1. git commit hash (short, 7 chars)
2. avg_loss achieved (e.g. 1.234567) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	avg_loss		status	description
a1b2c3d	0.997900		keep	baseline
b2c3d4e	0.993200		keep	increase LR to 0.04
c3d4e5f	1.005000		discard	switch to GeLU activation
d4e5f6g	0.000000		crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `neural-net-sandbox/mar-12`).

LOOP FOREVER:

1. Think of an experimental idea you would like to try and summarize it in a new experiment section in musings.md: explain the rationale behind it, explain how it connects to ML theory and anything else that is relevant to understand why you think this experiment is worth trying.
2. Implement your experiment idea in `train.py` or in `nets.py` by directly hacking the code and/or changing cmd line args values.
3. git commit
4. Run the experiment: `python.exe .\train.py --train [and any other cmd line args you choose] > run.log 2>&1` (redirect everything — do NOT let output flood your context)
5. Read out the results: `findstr /R "^Eval avg_loss:" run.log`
6. If the findStr output is empty, the run crashed. Run `powershell -Command "Get-Content run.log -Tail 50"` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If avg_loss improved (lower), you "advance" the branch, keeping the git commit
9.  If avg_loss is equal or worse, you git reset back to where you started
10. Immediately start the next iteration of this loop from step 1. Do not ask for permission. Do not stop to give a final summary unless the human explicitly asks you to pause or summarize.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Remember, you need execute a continuous stream of experiments.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
