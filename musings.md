# Experiment 000: Baseline

Rationale: establish the reference `Eval avg_loss` for the current defaults before changing anything. The current code already defaults to the `digit1h` representation, `TowersMLPForDigit1H`, `adamw`, batch size `1024`, and learning rate `0.0004`, so this run measures the incumbent recipe under the fixed 5 minute budget.

Expectation: this gives the loss threshold every later idea must beat. It also verifies that the branch and runtime are healthy before spending time on architectural edits.

Outcome: success. The run finished in-budget and reached `Eval avg_loss = 2.0453`, `Test avg_loss = 2.0317`. Training was still improving late in the budget, which suggests optimization and throughput changes are promising first directions before attempting larger architectural changes.

# Experiment 001: Time-Based Cosine LR Decay

Rationale: the baseline was still reducing training loss in the final seconds of the 5 minute window. With a constant learning rate, the optimizer may be ending the run before it ever gets a lower-noise refinement phase. A cosine decay tied to elapsed wall-clock time is a good fit for this setup because the budget is fixed in seconds, not in epochs. This should preserve aggressive early progress while giving the model a smaller step size near evaluation.

Theory connection: under a fixed compute budget, schedule shape matters as much as raw learning rate. Cosine decay is a standard way to spend most updates at a moderately high LR and then anneal toward a lower-variance regime, which often improves final validation loss without adding model complexity. A short warmup is included to avoid overly abrupt first updates.

Expectation: modest improvement over baseline if the current recipe is under-annealed rather than under-powered.

Outcome: success. `Eval avg_loss` improved to `1.9764` and `Test avg_loss` to `1.9626`, a clear gain over baseline with no architectural complexity added. This confirms the branch was under-annealed, so future experiments should keep the time-aware schedule and work from this checkpoint rather than reverting.

# Experiment 002: Higher Base Learning Rate

Rationale: after adding cosine decay, the train and eval losses became very close, which suggests the model is not obviously overfitting. That usually means there is room to push optimization harder. A modest LR increase should let the model cover more loss landscape early in the 5 minute window, while the cosine schedule still cools it down before evaluation.

Theory connection: with AdamW and a decay schedule, the base LR primarily controls the exploration rate in the high-noise phase. If the previous run remained stable and generalization stayed tight, increasing the base LR is a standard next move because it can improve both optimization speed and the final basin reached within a fixed budget.

Expectation: a small improvement if `0.0004` was conservative; otherwise likely a mild regression from overshooting.

Outcome: discard. The run remained stable and lowered training loss further, but the true metric slipped slightly to `Eval avg_loss = 1.9782`, which is worse than the current best `1.9764`. This suggests the higher early-step aggressiveness bought optimization progress on train without translating into better eval under the same 5 minute budget.

# Experiment 003: Lower Final LR Ratio

Rationale: the cosine schedule clearly helped, but the end-of-run losses were still moving when time expired. Since a higher base LR did not improve validation, the better direction may be to anneal more aggressively near the end rather than push harder at the beginning. Lowering the minimum LR ratio preserves the same early optimization behavior while giving the final chunk of the budget a smaller, less noisy step size.

Theory connection: late-stage annealing often matters most when train and eval losses are already tightly coupled. A lower terminal LR can let AdamW settle into a slightly better local basin or reduce oscillation around one, especially in small models trained under a short fixed horizon.

Expectation: a small gain if the current `0.1 * base_lr` floor is still too hot at the end of the run.

Outcome: success. `Eval avg_loss` improved to `1.9681` and `Test avg_loss` to `1.9653`. The gain is modest but real, and it reinforces the same story as Experiment 001: the best improvements so far come from spending the fixed budget with a calmer late optimization phase, not from increasing update aggressiveness.

# Experiment 004: Shorter Warmup

Rationale: the schedule improvements have all helped, but the current warmup still spends roughly the first 9 seconds of a 5 minute run ramping from near-zero LR. In a budget this short, that may be too conservative. Shortening warmup should recover some early optimization speed while preserving the now-proven benefits of a low final LR floor.

Theory connection: warmup is mainly there to avoid unstable first updates. Once training is already stable, excess warmup can become dead budget. Reducing warmup is a standard follow-up after a scheduled run behaves well, because it reallocates more of the compute budget to the productive high-LR phase.

Expectation: a small improvement if early under-training is the remaining bottleneck; otherwise no change or a slight regression from rougher startup dynamics.

Outcome: discard. Early training sped up substantially, but the true metric worsened to `Eval avg_loss = 1.9780`. That implies the extra early aggressiveness was not the missing ingredient; the best recipe so far benefits more from controlled late annealing than from shaving warmup time.

# Experiment 005: Lower Weight Decay

Rationale: with the current best schedule, train and eval losses remain tightly coupled, which makes heavy regularization look unnecessary. The current `adamw` decay may be damping useful late-stage parameter refinement more than it is preventing overfit. Reducing weight decay is the smallest way to test that hypothesis without changing architecture or optimizer family.

Theory connection: AdamW decouples shrinkage from the gradient step, so its weight decay acts as a direct prior toward smaller weights. When the model already shows little or no overfitting under a fixed compute budget, too much shrinkage can push the solution toward underfitting. Lowering it slightly can improve both optimization and final validation loss.

Expectation: a small gain if the current recipe is mildly over-regularized; otherwise no change or a slight regression from weaker inductive bias.

Outcome: discard. Training loss improved dramatically, but `Eval avg_loss` worsened to `1.9776`. That is a classic sign that the current recipe does still benefit from some shrinkage even though the raw train/eval gap is small, so the next moves should target representational bottlenecks instead of simply weakening regularization.

# Experiment 006: Deeper Aggregation Head

Rationale: the current tower compresses each 3-digit chunk, mixes the 3 chunks with attention, then uses a very small `192 -> 128 -> 1` head to produce the final regression output. That last stage may be the narrowest representational bottleneck in the whole model. A slightly deeper head is a targeted way to improve cross-chunk interaction capacity with much less runtime cost than widening the whole tower or attention block.

Theory connection: when most of the structure learning is already done upstream, the final MLP acts as the feature combiner. If it is too shallow, the model may underuse information already present in the chunk states. Adding one modest hidden layer can improve nonlinear feature composition without changing the training recipe or introducing major complexity.

Expectation: a small gain if final aggregation is the limiting stage; otherwise little change.

Outcome: discard. The deeper head was reasonably competitive at `Eval avg_loss = 1.9725`, but it still missed the current best `1.9681` while adding complexity. That weakens the bottleneck hypothesis enough that it is not worth keeping this version of the head.

# Experiment 007: Zero Final LR Floor

Rationale: the two clearest wins so far came from giving the optimizer a cooler endgame. Lowering the cosine floor from `0.1` to `0.02` helped, and neither higher LR nor shorter warmup did. The natural continuation of that signal is to let the schedule anneal all the way to zero by the end of the time budget.

Theory connection: in short fixed-horizon training, the terminal learning rate determines how much of the last slice of compute is spent refining versus still wandering. If the current floor remains slightly too hot, removing it entirely can turn the end of training into a true polishing phase. The risk is under-updating too early near the end.

Expectation: a small improvement if the best recipe still wants a colder finish; otherwise flat or slightly worse from over-annealing.

Outcome: discard. Fully annealing to zero reached lower training loss, but `Eval avg_loss` worsened to `1.9736`. This suggests the best point is not “as cold as possible”; the small nonzero floor at `0.02` seems to preserve just enough plasticity late in training to outperform the harder shutdown.

# Experiment 008: Increase Batch Size to 2048

Rationale: the current best recipe still improves steadily late into the 5 minute budget, and its per-epoch times suggest data-loader and optimizer overhead are nontrivial. A larger batch may improve hardware utilization and allow more wall-clock-efficient training, while also slightly reducing gradient noise during the later, colder part of the cosine schedule.

Theory connection: under a fixed time budget, the right batch size is not just a generalization question, it is a throughput question. If the model is currently under-updating because it spends too much time on optimizer step overhead, a larger batch can improve the quality reached in the same 300 seconds.

Expectation: improvement if throughput dominates; regression if the extra loss of gradient noise hurts exploration more than the speedup helps.

Outcome: discard. The larger batch improved throughput enough to reach many more epochs, but `Eval avg_loss` landed at `1.9699`, still slightly worse than the best `1.9681`. That suggests the speed/noise tradeoff might have a local sweet spot above `1024` but not as high as `2048`.

# Experiment 009: Increase Batch Size to 1536

Rationale: `2048` was slightly worse but close, which makes a smaller upward step worth testing. `1536` keeps some of the baseline noise profile while still reducing per-step overhead enough to potentially fit more useful optimization into the same time budget.

Theory connection: batch-size effects are often smooth rather than binary. A near-miss at a larger batch can indicate that the optimum sits somewhere in between, where throughput improves without washing out too much gradient stochasticity.

Expectation: a small gain if `2048` overshot the throughput/noise optimum only slightly.

Outcome: discard. `1536` regressed further to `Eval avg_loss = 1.9738`, so the batch-size search direction looks exhausted for now. The current best is likely benefiting from the original `1024` noise scale more than from extra throughput.

# Experiment 010: Increase Warmup Fraction to 0.05

Rationale: the shorter-warmup experiment failed, but that only tested one side of the curve. If the current recipe still burns some budget recovering from early high-variance updates, a slightly longer warmup could improve stability and leave the later cosine-decay phase in a better basin.

Theory connection: warmup length trades early progress against optimization stability. On adaptive optimizers, too little warmup can hurt even when the run does not visibly diverge, because the first few updates can still move parameters into a worse region. Testing the opposite direction is necessary before writing warmup off entirely.

Expectation: improvement only if the current start is still too abrupt; otherwise mild regression from spending too much budget underpowered.

Outcome: discard. A longer warmup produced `Eval avg_loss = 1.9717`, worse than the current best. Combined with the failure at `0.01`, this suggests the original `0.03` warmup is already close to the local optimum.

# Experiment 011: Switch Optimizer to Adam

Rationale: all wins so far were made with `AdamW`, but it is not clear whether the decoupled weight decay itself is helping or whether the schedule would work just as well, or better, with plain `Adam`. Since the current branch seems sensitive to regularization strength, changing optimizer family is worth testing before more detailed optimizer surgery.

Theory connection: `Adam` and `AdamW` differ mainly in how shrinkage is applied. On small problems and short budgets, that distinction can matter a lot because decoupled decay changes both the effective regularization and the step dynamics. Sometimes the simpler coupled update lands in a better short-horizon basin.

Expectation: likely neutral or slightly worse, but worth verifying because optimizer-family changes can dominate small schedule tweaks.

Outcome: discard. `Adam` regressed to `Eval avg_loss = 1.9729`, so the branch should stay on `AdamW`. The gains are therefore not just generic adaptive-optimizer effects; the current setup seems to benefit specifically from the decoupled weight-decay behavior.

# Experiment 012: Lower AdamW Beta2 to 0.95

Rationale: the training horizon is extremely short, so a very long second-moment memory may be suboptimal. Reducing `beta2` should make AdamW react faster to the current gradient scale, which can improve early and mid-run progress without changing the basic optimizer family that already outperformed plain Adam.

Theory connection: high `beta2` values smooth variance estimates over many steps, which is useful in long runs but can make the optimizer sluggish in short ones. Lowering `beta2` is a standard trick in compute-constrained training because it effectively increases optimizer responsiveness at the cost of noisier adaptation.

Expectation: possible improvement if the default `0.999` is too inertial for a 5 minute budget; otherwise regression from overreacting to noisy batches.

Outcome: discard. `Eval avg_loss` improved over plain Adam but still landed at `1.9710`, so faster variance adaptation did not beat the simpler default AdamW setup. Optimizer tuning looks lower-yield than schedule tuning on this problem.

# Experiment 013: Lower Base Learning Rate to 0.0003

Rationale: the branch has already shown that it likes a colder finish, and the higher-LR experiment was unhelpful. A small downward move in base LR tests the opposite possibility: the current recipe may still be a bit too aggressive throughout the run even with the better cosine floor.

Theory connection: when annealing changes help more than optimizer changes, it often means the effective step size is near the critical threshold. Nudging the base LR down can improve stability and final validation if the model currently oscillates around, rather than settles into, a good basin.

Expectation: improvement only if the best-known run is still slightly over-stepping; otherwise regression from slower progress.

Outcome: discard. Lowering the base learning rate to `0.0003` produced `Eval avg_loss = 1.9730`, which is clearly worse than the current best `1.9681`. The branch appears to prefer the existing `0.0004` step size, so further schedule work should focus on shape or architecture rather than globally slowing optimization.

# Experiment 014: Add Gradient Clipping at Norm 1.0

Rationale: the branch already appears close to a local optimum in terms of scalar schedule settings, so the remaining headroom may come from reducing rare but harmful update spikes rather than globally changing the learning rate. Gradient clipping is a targeted way to preserve the current recipe while suppressing occasional oversized steps.

Theory connection: adaptive optimizers do not eliminate gradient outliers; they mainly rescale them. In a short fixed-budget run, a few unstable updates can permanently waste optimization time by knocking the model out of a good basin. Norm clipping can improve validation even when train loss changes very little if the dominant issue is late-run instability.

Expectation: likely neutral to mildly positive if the best recipe is suffering from sporadic large updates; otherwise no effect.

Outcome: discard. Clipping at norm `1.0` produced `Eval avg_loss = 1.9711`, worse than `1.9681`. The run looked smoother in training, but the validation result says the baseline recipe is not being limited by rare gradient spikes in a way that clipping fixes.

# Experiment 015: Increase Attention Heads from 2 to 4

Rationale: the current architecture compresses the 9 digits into 3 chunk embeddings and then relies on a single `MultiheadAttention` layer to exchange information across those chunks. If the model is underusing cross-chunk structure, increasing the number of heads is a simple way to let it represent multiple interaction patterns without increasing sequence length or adding new blocks.

Theory connection: multi-head attention is most useful when different subspaces need to capture different relational cues. Even with only 3 tokens, a small number of heads can still bottleneck the diversity of learned interactions. Raising head count while keeping model width fixed mostly changes how the representational budget is partitioned, not the overall compute scale.

Expectation: small gain if chunk-to-chunk interaction diversity is currently the bottleneck; otherwise neutral or slightly worse from fragmenting the fixed 64-dimensional state.

Outcome: discard. Increasing the head count to `4` reached `Eval avg_loss = 1.9740`, worse than the best branch state. The lower training loss with worse eval suggests the original 2-head split is already adequate, and further attention complexity is more likely to overfit than help.

# Experiment 016: Widen the Post-Attention Feedforward Block

Rationale: if the branch is not bottlenecked by the attention pattern itself, it may still be bottlenecked by the tiny `64 -> 128 -> 64` feedforward subnetwork that processes each chunk after mixing. Widening that inner dimension is a targeted way to increase local nonlinear capacity without changing the training setup or the final head.

Theory connection: transformer-style residual blocks often derive a large part of their expressive power from the feedforward component, not only from attention. The FF block acts like a per-token feature refiner after relational information is exchanged. If its hidden width is too small, the model may fail to turn those mixed features into useful chunk representations.

Expectation: small gain if the current post-attention refinement is underpowered; otherwise neutral or slightly worse from extra overfitting.

Outcome: discard. Widening the feedforward block pushed `Eval avg_loss` to `1.9784`, substantially worse than the branch best. This is a strong sign that the current model is not capacity-limited in that block; adding width there mostly buys easier fitting of the training set.

# Experiment 017: Increase Digit Embedding Width from 24 to 32

Rationale: the model currently maps each one-hot digit into a 24-dimensional learned embedding before concatenating digits into 3 chunk vectors. If information is being compressed too aggressively at that earliest stage, later layers may never recover it. Increasing the embedding width is a clean way to test whether the front end is the real bottleneck.

Theory connection: representation bottlenecks near the input can be unusually damaging because every downstream block only sees the compressed features. A slightly wider embedding lets the model preserve more digit identity and positional interactions before chunk-level mixing begins, while still keeping the rest of the architecture intact.

Expectation: improvement if the existing 24-dimensional digit representation is too lossy; otherwise neutral or slightly worse from extra parameters.

Outcome: discard. The wider embedding reached `Eval avg_loss = 1.9681`, effectively tying the current best to reported precision, but it does so with a more complex front end. Under the simplicity criterion that is not enough to keep, so the branch should stay on the leaner original embedding width.

# Experiment 018: Reference Early Stop at 30 Epochs

Rationale: the full 5-minute run may be slightly overtraining relative to the evaluation metric. A short-horizon reference at `30` epochs tests whether much earlier stopping can buy better validation by spending less compute in the late, colder part of the cosine schedule.

Expectation: likely worse from undertraining, but useful as an anchor for the rest of the short-horizon screening sweep.

Outcome: discard. Stopping at `30` epochs yielded `Eval avg_loss = 2.0577`, far worse than the incumbent. The branch is nowhere near well-trained that early, so meaningful screening needs at least a moderately longer horizon.

# Experiment 019: Reference Early Stop at 45 Epochs

Rationale: `45` epochs is still materially shorter than the usual wall-clock-limited run while giving the model more time to settle than the `30`-epoch screen. This helps map where the undertraining regime begins to become competitive.

Expectation: better than `30` epochs but probably still worse than the full-budget incumbent.

Outcome: discard. `45` epochs improved materially over `30`, but `Eval avg_loss = 2.0204` is still nowhere close to `1.9681`. Early stopping this aggressively is not competitive.

# Experiment 020: Reference Early Stop at 60 Epochs

Rationale: `60` epochs is a reasonable compromise between signal quality and screening throughput. It serves as the main reference point for the large CLI sweep that follows.

Expectation: competitive enough for relative comparisons, but unlikely to beat the full-budget best.

Outcome: discard. The `60`-epoch reference landed at `2.0163`. That is still poor in absolute terms, but close enough to the `45`- and `90`-epoch runs to serve as a reasonable throughput-oriented screening anchor.

# Experiment 021: Reference Early Stop at 90 Epochs

Rationale: if the branch is only mildly overtraining by the end of the 5-minute run, a medium-length horizon may already recover most of the useful optimization while avoiding the least productive tail end.

Expectation: the strongest early-stop candidate in this first batch.

Outcome: discard. `90` epochs was the best early-stop variant at `1.9938`, but it still trails the incumbent by a wide margin. The current recipe benefits substantially from training past this horizon.

# Experiment 022: Reference Early Stop at 120 Epochs

Rationale: `120` epochs is still shorter than the wall-clock-limited default path but much closer to a mature optimization trajectory. This tests whether the best validation point may occur before the time budget is exhausted.

Expectation: the closest of the pure early-stop variants to the incumbent, though still unlikely to win outright.

Outcome: discard. `120` epochs regressed slightly to `2.0022` after the `90`-epoch improvement. That weakens the idea that there is a simple early-stop sweet spot materially before the time budget.

# Experiment 023: Lower Base LR to 0.00025 at 60 Epochs

Rationale: if short-horizon runs are effectively “hotter” because they spend less time in the low-LR tail, reducing the base learning rate may compensate and improve early validation.

Expectation: potentially stronger than the plain 60-epoch reference if the truncated run is otherwise too aggressive.

Outcome: discard. Lowering the LR this far collapsed short-horizon progress and produced `Eval avg_loss = 2.0552`. The truncated regime is not suffering from too much step size; if anything, it needs more urgency.

# Experiment 024: Lower Base LR to 0.0003 at 60 Epochs

Rationale: this revisits the lower-LR direction under a shorter compute horizon, where the tradeoff can differ from the full-budget result. A milder downward shift may suit early stopping better than long training.

Expectation: slightly better than the `0.00025` case if the 60-epoch regime still needs reasonable step size.

Outcome: discard. `0.0003` improved over `0.00025`, but `Eval avg_loss = 2.0240` was still worse than the plain 60-epoch reference. Lower LR remains the wrong direction in short training.

# Experiment 025: Lower Base LR to 0.00035 at 60 Epochs

Rationale: this is a small perturbation around the current best `0.0004`, aimed at finding whether a modestly cooler trajectory is optimal under reduced training length.

Expectation: one of the more plausible near-best short-horizon settings.

Outcome: discard. `0.00035` reached `2.0181`, almost identical to the 60-epoch reference and still well behind the incumbent. Mildly cooling the short run is not enough to rescue it.

# Experiment 026: Raise Base LR to 0.00045 at 60 Epochs

Rationale: early-stop regimes sometimes prefer slightly higher learning rates because they have fewer opportunities to accumulate progress. A small upward move tests that compensation effect.

Expectation: could outperform the lower-LR variants if the 60-epoch run is step-starved.

Outcome: discard. Raising the LR to `0.00045` produced `2.0244`, slightly worse than the base 60-epoch run. More aggression does not buy a better truncated trajectory here.

# Experiment 027: Raise Base LR to 0.0005 at 60 Epochs

Rationale: this pushes farther into the high-LR direction while staying closer to the incumbent than the already-discarded `0.0006` full-budget run. In short training, the stability boundary may move.

Expectation: probably too aggressive, but worth checking as the last LR screen in the first batch.

Outcome: discard. `0.0005` was the best of the LR-adjusted 60-epoch variants at `1.9955`, but it still lagged even the `90`-epoch reference. The main limitation is lack of optimization time, not a small LR misspecification.

# Experiment 028: Zero Warmup at 90 Epochs

Rationale: once the horizon is shortened to `90` epochs, even the existing `0.03` warmup consumes a meaningful fraction of useful updates. Removing warmup entirely tests whether medium-length training benefits from getting to full learning rate immediately.

Expectation: possible gain if the branch is under-updating early in the 90-epoch regime; otherwise worse from rougher startup dynamics.

Outcome: discard. Removing warmup completely produced `Eval avg_loss = 2.0812`, one of the worst medium-horizon results so far. The branch clearly still needs a gentle ramp even when training is truncated.

# Experiment 029: Shorter Warmup of 0.01 at 90 Epochs

Rationale: this is a softer version of the no-warmup test. If the medium-horizon regime wants slightly faster ramp-up but not a hard jump to full LR, a small warmup may be the compromise.

Expectation: better than zero warmup if the startup still needs some stabilization.

Outcome: discard. `0.01` warmup landed at `2.0806`, essentially matching the no-warmup failure. Medium-horizon training remains extremely sensitive to a too-abrupt start.

# Experiment 030: Longer Warmup of 0.05 at 90 Epochs

Rationale: the opposite direction is still worth checking because shorter runs can also be more sensitive to early instability. A bit more warmup may produce a cleaner basin even if it slows initial progress.

Expectation: likely worse, but it tests the stability side of the tradeoff directly.

Outcome: discard. A longer warmup improved sharply over the shorter-warmup failures, but `2.0376` was still far worse than the 90-epoch reference. Stability matters, but extra warmup alone does not make the shorter horizon competitive.

# Experiment 031: Zero Final LR Floor at 90 Epochs

Rationale: the full-budget branch preferred a nonzero floor, but that may change when training stops much earlier. In the `90`-epoch regime, letting the schedule cool harder by the end may improve validation if late updates are still too noisy.

Expectation: possible medium-horizon gain despite the full-budget miss.

Outcome: discard. Fully annealing to zero again hurt badly, reaching `2.0626`. The medium-horizon regime also prefers retaining some late-step plasticity.

# Experiment 032: Lower Final LR Floor to 0.01 at 90 Epochs

Rationale: this is a less extreme variant of the zero-floor test. If the 90-epoch run wants a colder finish than `0.02` but not a full shutdown, `0.01` is the natural interpolation.

Expectation: one of the more plausible schedule-shape improvements in this batch.

Outcome: discard. Lowering the floor modestly to `0.01` gave `2.0167`, which is still much worse than the 90-epoch reference. The shorter horizon does not want a colder finish either.

# Experiment 033: Raise Final LR Floor to 0.05 at 90 Epochs

Rationale: the medium-horizon regime might instead benefit from staying more plastic through its shorter training window. A higher LR floor tests whether the branch is prematurely freezing under the default setting.

Expectation: improvement only if the truncated run still needs meaningful motion late in training.

Outcome: discard. Raising the floor to `0.05` produced `2.0010`, the best schedule-shape result of this batch but still clearly behind the incumbent. If the 90-epoch regime wants anything, it wants to stay somewhat warmer late, though not enough to matter globally.

# Experiment 034: Remove Weight Decay at 90 Epochs

Rationale: shorter runs often overfit less simply because they fit less. Removing weight decay tests whether the 90-epoch regime is now under-regularized by compute alone and wants more freedom to reduce error.

Expectation: could beat the 90-epoch reference if regularization is the main remaining brake on progress.

Outcome: discard. Removing weight decay reached `1.9965`, the best result in this batch, which supports the idea that shorter runs are less in need of regularization. Even so, the gain is far too small to make medium-horizon training competitive with the full-budget branch.

# Experiment 035: Lower Weight Decay to 0.00005 at 90 Epochs

Rationale: this is a milder regularization reduction than outright removal. It targets the same hypothesis while retaining some shrinkage signal that the full-budget branch still seemed to like.

Expectation: more plausible than zero decay if the best point is only slightly under-regularized at 90 epochs.

Outcome: discard. Lowering weight decay to `0.00005` gave `2.0024`, worse than both zero decay and the reference. The medium-horizon effect is not a smooth small-decay optimum.

# Experiment 036: Raise Weight Decay to 0.0002 at 90 Epochs

Rationale: if the medium-horizon regime is landing in noisy or brittle basins, slightly stronger regularization could help its validation metric even without longer training.

Expectation: likely neutral to worse, but worth testing against the lighter-decay direction.

Outcome: discard. Stronger regularization at `0.0002` landed at `2.0337`, confirming that the shorter regime already has enough implicit regularization from undertraining.

# Experiment 037: Raise Weight Decay to 0.0005 at 90 Epochs

Rationale: this extends the weight-decay sweep farther into the strong-regularization side. It tests whether the 90-epoch regime prefers a much smoother, more biased solution over a loosely fit one.

Expectation: probably too strong, but it closes out the medium-horizon decay band cleanly.

Outcome: discard. `0.0005` weight decay collapsed to `2.1223`, the worst result in the batch. Strong explicit shrinkage is decisively incompatible with the 90-epoch regime.

# Experiment 038: Reduce Batch Size to 512 at 60 Epochs

Rationale: smaller batches inject more gradient noise and can sometimes improve validation in short training by exploring better basins early. This tests whether the 60-epoch regime benefits from more stochasticity.

Expectation: possibly better than the 60-epoch reference if it is currently too deterministic.

Outcome: discard. Halving the batch size gave `Eval avg_loss = 2.0183`, barely different from the 60-epoch reference. Extra gradient noise alone does not materially improve short-horizon training.

# Experiment 039: Reduce Batch Size to 768 at 60 Epochs

Rationale: this is a milder move toward more noise and more optimizer steps without the full throughput penalty of halving batch size.

Expectation: a more plausible batch-size sweet spot than `512`.

Outcome: discard. `768` reached `2.0238`, worse than both the `512` run and the 60-epoch reference. There is no evidence of a helpful smaller-batch sweet spot here.

# Experiment 040: Increase Batch Size to 1536 at 60 Epochs

Rationale: larger batches reduce gradient noise and improve wall-clock efficiency per epoch. Under a short explicit epoch cap, that mostly tests whether a smoother optimization path helps more than extra stochasticity.

Expectation: likely neutral to slightly worse, but informative relative to the smaller-batch screens.

Outcome: discard. `1536` produced `2.0190`, almost identical to the smaller-batch runs. Moderate large-batch smoothing does not fix the short-horizon regime either.

# Experiment 041: Increase Batch Size to 2048 at 60 Epochs

Rationale: this extends the large-batch direction farther, emphasizing smoother updates over noise. It also checks whether the 60-epoch regime prefers cleaner gradients even at lower update count per sample.

Expectation: probably worse unless the model is highly noise-limited early on.

Outcome: discard. `2048` regressed sharply to `2.0764`. Pushing the short run into a low-noise regime is clearly harmful.

# Experiment 042: Increase Batch Size to 3072 at 60 Epochs

Rationale: a very large batch sharply reduces step noise and changes the optimization character of the short run. This is a stress test of the large-batch hypothesis.

Expectation: mostly a negative control.

Outcome: discard. `3072` collapsed further to `2.1299`, reinforcing that very large batches are fundamentally misaligned with the short screening horizon.

# Experiment 043: Increase Batch Size to 4096 at 60 Epochs

Rationale: this pushes the large-batch sweep to an extreme that may expose whether the short-horizon regime is fundamentally limited by stochasticity rather than compute.

Expectation: likely poor, but it closes the batch-size direction cleanly.

Outcome: discard. `4096` remained terrible at `2.1055`. Extremely large batches are conclusively ruled out for this truncated regime.

# Experiment 044: Batch Size 512 with Lower LR 0.0003 at 60 Epochs

Rationale: if a smaller batch helps via noise but destabilizes via too-large effective steps, lowering LR alongside it may recover the good part of that tradeoff.

Expectation: more plausible than raw `512` batch size if instability is the limiting factor.

Outcome: discard. Pairing `512` with a lower LR reached `2.0146`, slightly better than plain `512` but still poor overall. The smaller-batch regime is not being rescued by cooling it down.

# Experiment 045: Batch Size 512 with Higher LR 0.0005 at 60 Epochs

Rationale: the opposite coupling is also worth testing because small batches can tolerate, or even prefer, slightly larger learning rates when the goal is fast early progress.

Expectation: likely noisy, but it tests the aggressive small-batch corner directly.

Outcome: discard. This was the best batch-size screen at `1.9993`, suggesting that if a short run wants anything, it wants the combination of more updates and slightly more aggression. Even so, it remains far behind the full-budget incumbent.

# Experiment 046: Batch Size 2048 with Higher LR 0.0005 at 60 Epochs

Rationale: large batches often want a somewhat larger LR to compensate for reduced noise. This tests a simple scaled-up pairing rather than batch size alone.

Expectation: better than plain `2048` if the base LR underdrives the large-batch regime.

Outcome: discard. Raising LR for the `2048` batch only improved it to `2.0573`, still very poor. The large-batch failure is not just a simple LR mismatch.

# Experiment 047: Batch Size 2048 with Higher LR Floor 0.05 at 60 Epochs

Rationale: another way to compensate for large-batch smoothness is to stay warmer later in training instead of only raising the base LR. This probes that interaction directly.

Expectation: modestly better than plain `2048` if the large-batch path freezes too early.

Outcome: discard. A warmer late phase with `2048` batch yielded `2.0395`, better than plain `2048` but nowhere near enough. Large-batch short training remains a dead end.

# Experiment 048: Adam with Constant LR at 60 Epochs

Rationale: a truncated run may benefit from avoiding cosine decay entirely if it needs to keep moving aggressively throughout. Plain Adam with constant LR is the simplest alternate optimizer dynamic to test.

Expectation: likely worse than AdamW full-budget training, but informative about whether decay is too restrictive in short runs.

Outcome: discard. Adam with constant LR reached `2.0563`, much worse than the AdamW screens. The short run still benefits from explicit schedule shaping.

# Experiment 049: Adam with Cosine LR at 60 Epochs

Rationale: this isolates the effect of optimizer family from the schedule by reusing the current cosine shape with plain Adam.

Expectation: potentially competitive with the short-horizon AdamW references, but unlikely to beat them.

Outcome: discard. Adam with cosine reached `2.0136`, which is better than constant Adam but still not competitive even within the short-horizon regime. AdamW remains the stronger adaptive family here.

# Experiment 050: RMSprop with Constant LR at 60 Epochs

Rationale: RMSprop can behave differently from Adam-family methods in short adaptive training, especially on regression-style objectives. A constant-LR screen checks its raw early optimization character.

Expectation: probably poor, but worth ruling out explicitly.

Outcome: discard. Constant-LR RMSprop was very poor at `2.4405`. The family is not naturally aligned with this sandbox.

# Experiment 051: RMSprop with Cosine LR at 60 Epochs

Rationale: if RMSprop has any value here, it is more likely to appear with the same broad schedule shape that helped AdamW.

Expectation: better than constant RMSprop if the family is usable at all.

Outcome: discard. Cosine scheduling made RMSprop even worse at `2.5390`. This direction can be closed entirely.

# Experiment 052: Adagrad with Constant LR at 60 Epochs

Rationale: Adagrad front-loads adaptation and can sometimes excel in short runs with sparse or structured inputs. This tests that extreme adaptive bias directly.

Expectation: highly uncertain, but potentially interesting in one-hot digit space.

Outcome: discard. Adagrad with constant LR reached `2.1649`, poor despite the one-hot input structure. Its front-loaded adaptation is not enough to compensate.

# Experiment 053: Adagrad with Cosine LR at 60 Epochs

Rationale: combining Adagrad with cosine decay checks whether its aggressive early adaptation benefits from a cleaner finish.

Expectation: likely similar to or slightly better than constant Adagrad if the family is viable.

Outcome: discard. Cosine Adagrad improved to `2.1222`, but that is still far off the useful range. Adagrad is not a serious contender here.

# Experiment 054: SGD with Constant LR at 60 Epochs

Rationale: while SGD is unlikely to dominate adaptive methods here, a short-horizon constant-LR baseline is useful as a lower-complexity control.

Expectation: poor, but it closes out the optimizer family space.

Outcome: discard. Constant-LR SGD stalled at `7.6140`, confirming that non-adaptive optimization is dramatically underpowered in this setup and horizon.

# Experiment 055: SGD with Cosine LR at 60 Epochs

Rationale: if SGD can work at all in this sandbox, a cosine schedule is more likely to stabilize it than a constant rate.

Expectation: slightly better than constant SGD, still probably far from competitive.

Outcome: crash. SGD with cosine decay diverged to `nan` before evaluation settled, so it is not just uncompetitive but unstable under this configuration.

# Experiment 056: Adam with Cosine LR 0.0005 and No Weight Decay at 60 Epochs

Rationale: the medium-horizon sweep hinted that shorter runs may prefer less regularization. This gives Adam its friendliest plausible short-run setting in one targeted check.

Expectation: one of the stronger non-AdamW candidates in this batch.

Outcome: discard. This tuned Adam variant reached `2.0122`, nearly matching plain Adam with cosine but still nowhere near the incumbent. Even its friendliest short-run setting does not justify switching away from AdamW.

# Experiment 057: RMSprop with Cosine LR 0.0003 and No Weight Decay at 60 Epochs

Rationale: this is a tuned RMSprop attempt rather than a default-like one. Lower LR and no explicit decay give it a better chance to show whether the family deserves more attention.

Expectation: better than the raw RMSprop screens if the family has any usable niche here.

Outcome: discard. Tuned RMSprop still failed badly at `2.5347`. The optimizer-family search is now decisively exhausted.

# Experiment 058: Normalized-Int Input with AdamW Cosine at 60 Epochs

Rationale: most of the session has focused on the `digit1h` tower architecture. It is worth checking whether the simpler scalar input path is actually better aligned with short-horizon optimization under the improved training recipe.

Expectation: likely worse, but it provides a clean representational control.

Outcome: discard. The scalar normalized-int path was extremely poor at `6.2337`. It is not remotely competitive with the current digit-one-hot tower under the modern training recipe.

# Experiment 059: Digit Input with AdamW Cosine at 60 Epochs

Rationale: raw digit inputs preserve per-position structure without the expansion cost of one-hot encoding. This may help shorter runs if the larger one-hot/tower model is too heavy for the horizon.

Expectation: the most plausible non-`digit1h` representation in this final sweep.

Outcome: discard. Raw digits with AdamW cosine reached `2.4549`, far better than the scalar or binary paths but still nowhere near useful. The simpler digit MLP is not close to the tower model.

# Experiment 060: Binary Input with AdamW Cosine at 60 Epochs

Rationale: binary features expose arithmetic structure differently from decimal digits. Even if unlikely, they are cheap to test and may favor shorter training.

Expectation: probably poor, but worth ruling out explicitly.

Outcome: discard. Binary inputs landed at `5.9458`, confirming that this representation is a dead end under short-horizon training.

# Experiment 061: Normalized-Int Input with AdamW Cosine at 90 Epochs

Rationale: if the scalar path is simply slower to optimize than `digit1h`, giving it a somewhat longer horizon may be necessary for a fair comparison.

Expectation: better than the 60-epoch scalar run, still likely inferior overall.

Outcome: discard. Giving the scalar path more training time did not help; it worsened to `6.3246`. This model family is fundamentally mismatched.

# Experiment 062: Digit Input with AdamW Cosine at 90 Epochs

Rationale: this gives the raw-digit MLP its best chance in the final sweep. If it is competitive at all, it should start to show up here rather than at 60 epochs.

Expectation: the strongest alternative-representation candidate in the last batch.

Outcome: discard. Raw digits improved to `2.3869` at `90` epochs, but that is still vastly worse than the incumbent. Even the best non-`digit1h` representation is not in the same league.

# Experiment 063: Binary Input with AdamW Cosine at 90 Epochs

Rationale: this mirrors the 90-epoch fairness check for binary inputs. It tests whether the binary path was simply undertrained at 60 epochs.

Expectation: still likely weak.

Outcome: discard. Binary inputs remained unusable at `5.9211`. Extra training time does not rescue this path.

# Experiment 064: Normalized-Int Input with Adam Constant LR at 60 Epochs

Rationale: the scalar path may prefer a simpler optimizer dynamic than the tower model. Plain Adam with constant LR is the cheapest alternative to test.

Expectation: probably worse than AdamW cosine, but it checks one plausible interaction.

Outcome: discard. Constant-LR Adam on the scalar path reached `6.3647`, even worse than the already-bad AdamW scalar screen. There is no optimizer trick hiding here.

# Experiment 065: Digit Input with Adam Constant LR at 60 Epochs

Rationale: if the raw-digit MLP is a simpler model, it may not need schedule complexity to the same degree as `digit1h`. This tests that directly.

Expectation: possibly competitive with its AdamW-cosine counterpart, though unlikely to exceed it.

Outcome: discard. Raw digits with constant-LR Adam gave `2.4794`, slightly worse than the AdamW-cosine version. The digit MLP does not benefit from removing schedule complexity.

# Experiment 066: Binary Input with Adam Constant LR at 60 Epochs

Rationale: this closes the representation-by-optimizer interaction space with the same simple Adam constant-LR rule.

Expectation: poor, but it completes the grid cleanly.

Outcome: discard. Constant-LR Adam on binary inputs reached `5.8421`, still extremely poor. This closes the binary path completely.

# Experiment 067: Normalized-Int Input with AdamW Constant LR at 90 Epochs

Rationale: one final scalar-path variant checks whether longer training plus a constant step size is a better match than cosine decay for the normalized-int MLP.

Expectation: unlikely to matter, but it closes the requested fifty-experiment extension with a distinct representational hypothesis.

Outcome: discard. The final scalar-path variant finished at `6.3410`, confirming beyond doubt that the normalized-int family is not competitive. The session extension ends with the original `digit1h` tower still overwhelmingly dominant.

# Experiment 068: Widen the Chunk Tower Hidden Width to 320

Rationale: the current per-chunk MLP may still be the narrowest computational bottleneck before attention has a chance to mix information. A moderate width increase tests whether richer chunk-local features improve final accuracy more than they overfit.

Expectation: small improvement if the chunk encoder is underpowered; otherwise neutral to slightly worse.

Outcome: discard. Increasing the chunk-tower width to `320` produced `Eval avg_loss = 1.9736`, worse than the incumbent. More local capacity by itself does not improve the branch.

# Experiment 069: Narrow the Chunk Tower Hidden Width to 192

Rationale: the branch has repeatedly shown a preference for simpler models over extra capacity. A narrower tower tests whether the current `256` width is already more than necessary and whether a smaller encoder generalizes better within the fixed budget.

Expectation: possible simplification win if the current tower is mildly over-parameterized.

Outcome: discard. Narrowing the tower to `192` was competitive at `1.9721`, but still worse than the current best. The branch does not appear overbuilt enough for this simplification to pay off.

# Experiment 070: Deepen the Chunk Tower to 3 Hidden Layers

Rationale: if chunk encoding needs more nonlinear composition rather than just more width, an extra hidden layer is the most direct way to test it. This keeps the attention and head unchanged while increasing local depth.

Expectation: modest gain only if chunk-level feature hierarchy is currently too shallow.

Outcome: keep. The 3-layer chunk tower improved validation to `1.9664`, beating the prior best `1.9681`. After materializing the same architecture as the default code path and rerunning from a real code commit, it improved further to `1.9629` on commit `76f2004`, which becomes the new branch base.

# Experiment 071: Shallow the Chunk Tower to 1 Hidden Layer

Rationale: the opposite depth move checks whether the current tower is doing unnecessary transformation before attention. If the task is simpler than the model assumes, removing one layer may improve optimization and generalization.

Expectation: possible simplification win if the current chunk encoder is over-processing the input.

Outcome: discard. Reducing the tower to a single hidden layer regressed to `1.9729`. The current model does need substantial chunk-local transformation before mixing.

# Experiment 072: Widen Chunk States to 96 with 3 Attention Heads

Rationale: the current `64`-dimensional chunk state may compress inter-chunk information too aggressively. Increasing chunk width while matching the attention head count to `3` gives the mixer more representational room without changing sequence length.

Expectation: improvement if the main bottleneck is chunk-state bandwidth rather than optimizer behavior.

Outcome: discard. The wider `96`-dimensional chunk state reached only `1.9758`, so the branch is not obviously bandwidth-limited in the mixer. Extra state size mostly adds complexity without helping validation.

# Experiment 073: Narrow Chunk States to 48 with 3 Attention Heads

Rationale: a smaller chunk state tests the opposite hypothesis: the model may benefit from a tighter inductive bias and less capacity in the attention block. This is a structural simplification rather than a regularization tweak.

Expectation: either a mild simplification win or a clear regression if the current state width is already close to minimal.

Outcome: discard. The narrower `48`-dimensional chunk state reached `1.9669`, which is respectable but still behind the new deeper-tower base. The model can be compressed somewhat, but not enough to improve the metric.

# Experiment 074: Switch the Tower Activation to GeLU

Rationale: SiLU has worked well so far, but GeLU is often stronger in transformer-like blocks because of its smoother gating behavior. This isolates activation geometry from width and depth changes.

Expectation: small gain if smoother nonlinear gating matters in the tower and head.

Outcome: discard. GeLU landed at `1.9688`, slightly worse than the current branch best. The existing SiLU activation still seems better aligned with this architecture.

# Experiment 075: Switch the Tower Activation to ReLU

Rationale: ReLU is a simpler nonlinearity with stronger sparsity bias than SiLU. If the current branch is over-smoothing representations, ReLU may produce cleaner features and better generalization.

Expectation: likely worse, but worthwhile as a simpler activation control.

Outcome: discard. ReLU was surprisingly competitive at `1.9657`, but it still missed the best result. That suggests activation choice matters, but not enough to justify switching away from SiLU yet.

# Experiment 076: Remove Digit Position Embeddings

Rationale: the 9-digit order is already implicit in the fixed chunking and input layout. Learned digit-position embeddings may therefore be redundant or even slightly harmful noise.

Expectation: possible simplification win if local position is already encoded well enough by structure alone.

Outcome: discard. Removing digit position embeddings regressed to `1.9711`, indicating that local position information is still important even with fixed chunking.

# Experiment 077: Remove Chunk Position Embeddings

Rationale: with only 3 chunks and a flattening head, the network may already know chunk identity from feature order alone. Removing chunk-position embeddings tests whether they are useful signal or just extra parameters.

Expectation: likely neutral to mildly positive if chunk order is already easy to infer.

Outcome: keep. Dropping chunk position embeddings improved validation to `1.9644`, beating the deeper-tower base. After materializing that change as the default code path and rerunning from a real code commit, it improved further to `1.9626` on commit `12984b9`, which becomes the new branch base.

# Experiment 078: Remove Cross-Chunk Attention

Rationale: the branch has not clearly shown that attention capacity is the limiting factor, and some earlier attention expansions hurt. This ablation tests whether the attention block is helping at all beyond what the final head can recover from independent chunk encodings.

Expectation: probably worse, but highly informative about where the real modeling gain comes from.

Outcome: discard. Removing attention regressed to `1.9683`, so the cross-chunk mixer is still doing meaningful work even in the simplified architecture.

# Experiment 079: Remove the Post-Attention Feedforward Block

Rationale: the FF residual sub-block may be redundant once attention has mixed the chunk states. Removing it tests whether the architecture can generalize better with a leaner transformer-style block.

Expectation: possible simplification win if the FF block mostly adds fitting capacity rather than useful refinement.

Outcome: keep. Dropping the FF block improved validation to `1.9615`, a clean simplification win. After materializing the change in code and rerunning from a real commit, the branch improved again to `1.9620` on `a5aafb0`, which becomes the new best base.

# Experiment 080: Remove Both Attention and Feedforward Blocks

Rationale: this is the full mixer ablation, reducing the architecture to chunk encoding plus a final head. It tests whether most of the performance comes from local chunk features and the terminal combiner rather than token mixing.

Expectation: clear regression, but it defines the lower bound of the mixer stack’s value.

Outcome: discard. Removing both mixer sub-blocks collapsed performance to `2.0957`, confirming that attention is the essential part and the FF block was the expendable one.

# Experiment 081: Use Mean Pooling with a Deeper Head

Rationale: flattening preserves chunk order but may force the final head to learn invariances inefficiently. Mean pooling trades order information for a simpler aggregated representation; a deeper head compensates by adding nonlinear capacity after pooling.

Expectation: improvement only if the current flatten-and-MLP head is an awkward way to aggregate the 3 chunk states.

Outcome: discard. Mean pooling with a deeper head reached only `1.9708`. The architecture still benefits from preserving explicit chunk order into the final head.

# Experiment 082: Deepen the Final Head to 2 Hidden Layers

Rationale: the prior deeper-head experiment was close but not enough to keep. Retesting head depth through the new configurable path is still worthwhile because it remains one of the cleaner ways to increase cross-chunk feature composition.

Expectation: competitive but unlikely to beat the incumbent without a larger supporting signal.

Outcome: discard. A deeper final head again proved competitive but not good enough at `1.9678`. The current branch prefers the simpler head once the chunk tower is stronger and the FF block is gone.

# Experiment 083: Widen the Final Head Hidden Width to 192

Rationale: if the final head is the only real aggregator after chunk mixing, it may benefit more from width than from depth. This targets head capacity with minimal architectural disruption.

Expectation: small gain if the current `128`-unit head is too narrow.

Outcome: discard. Widening the head to `192` gave `1.9666`, which is respectable but clearly worse than the simplified mixer base. The head is not the main remaining bottleneck.

# Experiment 084: Reduce Digit Embedding Width to 18

Rationale: a narrower digit embedding tests whether the early representation can be simplified without harming the downstream tower. This is the opposite of the earlier widened-embedding near-tie.

Expectation: either a small simplification win or a noticeable drop if the current front-end width is already near the floor.

Outcome: discard. Reducing digit embeddings to `18` reached `1.9655`, which is close but still worse than the branch best. The front end can be tightened somewhat, but not enough to improve the metric.

# Experiment 085: Reduce Digit Embedding Width to 18 with Wider Head 160

Rationale: if narrowing the digit embedding removes a bit too much capacity, a slightly wider final head may recover it while still shifting the overall architecture toward a leaner front end. This tests whether the best capacity tradeoff is “smaller embeddings, stronger aggregator.”

Expectation: improvement only if the narrower front end needs mild compensation at the output stage.

Outcome: discard. Pairing `digit_embed_dim = 18` with a `160`-wide head landed at `1.9681`, worse than both the simpler `18`-dimensional front end and the current best. The extra head width does not recover enough signal to justify the added complexity.

# Experiment 086: Use Digit Embedding 30 with Chunk Dim 60

Rationale: this shifts capacity toward the front end while keeping chunk states modest. It tests whether richer per-digit encoding plus a leaner mixer is a better allocation than the current `24 -> 64` shape.

Expectation: improvement only if the front-end compression is more limiting than the chunk-state size.

Outcome: discard. Reallocating capacity to `30 -> 60` produced `1.9660`, which is competitive but still behind the branch best. This capacity shift does not beat the cleaner current allocation.

# Experiment 087: Use Chunk Dim 80 with 5 Attention Heads

Rationale: this is a balanced wider-state variant that keeps attention heads small relative to channel count. It explores a different capacity allocation from the earlier 96-dimensional state test.

Expectation: one of the more plausible wider-mixer variants if the current chunk state is the bottleneck.

Outcome: discard. The `80`-dimensional, 5-head mixer regressed to `1.9696`. Wider mixer states are not helping once the simpler structural cleanups are in place.
