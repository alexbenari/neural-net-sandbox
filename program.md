# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `neural-net-sandbox/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b neural-net-sandbox/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `data folder` contains the train, val and test sets. Do not modify.
   - `nets.py, train.py` — the files you modify. Model architecture, optimizer, training loop.

3. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
4. **Go: Kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **limited time budget of 5 minutes at most**. You launch it simply as: `python.exe .\train.py --train`. The baseline (i.e. first) run uses the default cmd line param values. 

**What you CAN do:**
- Modify `train.py` and/or 'nets.py' — these are the only files you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc. Many of these can be modified via command line args (e.g. learning rate, num epochs and others). If you want to change hyperparams that are controlled via command line parameters, use the cmd line arg. You can also decide to add more cmd line args. Try to restrict code changes to things like the structure of the MLP and similar ideas which do not naturally lend themselves to being expressed as simple cmd line args. 

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading and loss function definition code.
- Install new packages or add dependencies.
- Modify the evaluation harness. The avg_loss metric returned by the evaluate function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest avg_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always at most 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful abg_loss improvement, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 avg_loss improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 avg_loss improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is:
python.exe .\train.py --train

## Output format

Once the script finishes it prints a summary like this:

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

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

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

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code and/or changing cmd line args values.
3. git commit
4. Run the experiment: `python.exe .\train.py --train [and any other cmd line args you choose] > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `findstr /R "^Eval avg_loss:" run.log`
6. If the grep output is empty, the run crashed. Run `powershell -Command "Get-Content run.log -Tail 50"` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If avg_loss improved (lower), you "advance" the branch, keeping the git commit
9. If avg_loss is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
