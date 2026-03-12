# Musings

## Structured chunk features for the `digit1h` tower
The target function is the character length of English number words, so it has obvious structure at the 3-digit chunk level: whether a chunk is zero, whether it is below 100, and whether the final chunk triggers the cross-chunk `"and"` pattern all matter. The baseline tower already splits the input into three 30-wide one-hot chunks, so the idea here was to expose a few of those handpicked structural flags directly to the head instead of forcing the network to rediscover them from scratch.

From an ML perspective, this was an inductive-bias experiment. If the hand-designed features match the true compositional structure, they should reduce sample complexity and help generalization. The risk was that I would inject the wrong abstraction boundary and encourage the model to latch onto brittle heuristics; that is exactly what the result suggested, because train loss improved while eval loss got worse.

## Adam learning rate `5e-4`
Once the branch baseline was encoded in code and replayed well, the next obvious question was whether the current setup was simply under-optimizing within the fixed 300-second budget. The model only reaches about 250 epochs before the timer hits, so a modestly larger Adam step size looked like the fastest way to test whether we could trade some stability for faster descent without rewriting the model.

This is basically a time-budget allocation question. With a hard wall-clock cap, the best hyperparameters are not the ones that would win after full convergence but the ones that spend the first few hundred optimizer steps most effectively. In practice `5e-4` was too aggressive: both train and eval got worse, so the baseline seems to sit closer to the stable side of the useful learning-rate range.

## Batch size `2048`
After the higher learning-rate miss, I tested the other obvious wall-clock lever: batch size. The thought was that doubling the batch might buy more epochs and a cleaner gradient estimate within the same 300 seconds, which sometimes helps smooth regression problems when the architecture is already good enough and optimization noise is the real bottleneck.

The theoretical tradeoff is classic optimization versus generalization. Larger batches reduce gradient noise and can improve hardware efficiency, but they also reduce the number of parameter updates per unit of data and often hurt the implicit regularization that smaller batches provide. The result was clearly worse on eval, which argues that the current model benefits more from the noisier `1024`-sized updates than from the extra throughput of `2048`.

## SiLU activations in the `digit1h` tower and head
The current best model still looks optimization-limited rather than capacity-limited: it makes steady progress for the full 300 seconds, but the loss curve stays noisy and never looks fully settled. ReLU is a fine default, but on a smooth scalar regression target it can be worth testing a smoother nonlinearity like SiLU, especially when the inputs are dense hidden activations coming off several linear layers rather than raw image-like tensors.

The ML-theory intuition is modest but real. SiLU preserves gradient flow better around zero and can fit smooth corrections with fewer sharp kinks, which sometimes improves both optimization stability and out-of-distribution behavior in MLPs. This is a narrow architecture change, so it is a good next probe after the failed larger structural edits.

## Slightly higher Adam LR on top of SiLU
Now that SiLU has actually improved eval, the most natural follow-up is to revisit the learning-rate question in the new activation regime instead of assuming the old ReLU result still applies unchanged. The smoother nonlinearity may tolerate a somewhat larger step size, and the SiLU run also completed fewer epochs within the same wall-clock budget, so there is a plausible case that a modest LR increase could recover some of that lost optimization speed.

This is still a conservative optimizer probe, not a blind sweep. I already know `5e-4` was too much for the ReLU baseline, so the point is not to jump back to that level blindly but to test whether the stability region shifted after the architectural improvement. If it did, the interaction between activation smoothness and step size could be worth more than another isolated hyperparameter tweak.

## Tiny AdamW regularization on top of the SiLU baseline
The new SiLU baseline is good enough that the next sensible question is whether generalization can be nudged with a very small amount of explicit regularization instead of more capacity or a more aggressive optimizer. The previous AdamW trial was on the old ReLU setup and used a much larger weight decay, so it does not really answer the narrower question of whether the improved architecture benefits from a light decoupled penalty.

This is a standard bias-variance probe. In overparameterized MLPs, a tiny `weight_decay` can sometimes trim just enough parameter drift to improve eval without materially slowing early optimization, especially once the base optimizer and activation are already in a good regime. I am keeping this deliberately small so the test isolates regularization rather than turning into a full optimizer swap.

## Slightly stronger weight decay around the new AdamW baseline
Once a tiny amount of decoupled weight decay helped, the right next move is not a big jump but a local sweep around that value. The current best is `1e-4`, so trying `2e-4` is a cheap way to check whether we are on the rising side of the regularization curve or already past the optimum.

This is mostly about calibrating the implicit versus explicit regularization balance. The SiLU + AdamW setup is already fitting the training set well, and a slightly stronger penalty can sometimes improve out-of-distribution behavior on these small tabular-style problems. If it degrades, that is still useful because it brackets the sweet spot more tightly.

## Moderate batch-size reduction with the SiLU + AdamW baseline
The current baseline still runs with a fairly large batch of `1024`, which is good for throughput but may be washing out some of the gradient noise that helps generalization. I already know that going larger to `2048` was bad; the more interesting open question is whether moving somewhat smaller helps now that the optimizer and activation are in a better regime.

From an optimization point of view, this is testing a different noise scale rather than a different model. Smaller batches usually mean more stochastic updates and a slightly different implicit regularization effect, which can matter a lot on small structured regression tasks. `768` is a moderate move, so it is unlikely to be a pure hardware-efficiency disaster and should still tell me whether the current optimum prefers noisier updates.

## Slightly weaker weight decay around the AdamW sweet spot
The `2e-4` run suggested that the useful regularization window is not above `1e-4`, but that still leaves open the possibility that the true optimum sits slightly below it. A test at `5e-5` is the natural complement because it tightens the bracket from the other side without changing anything else about the optimizer or model.

This is a local hyperparameter-identification step rather than a new modeling hypothesis. On noisy single-run evaluations, it is useful to understand the shape of the neighborhood around a good setting instead of assuming the first improvement is the optimum. If `5e-5` is also worse, then `1e-4` starts to look like a real attractor rather than a lucky point estimate.

## Slightly lower learning rate inside the AdamW baseline
With weight decay bracketed, the next local degree of freedom is step size. The best kept AdamW run used `lr=4e-4`, but single-run noise is now large enough that it is worth checking whether a slightly smaller step gives better stability in the same neighborhood instead of assuming the current value is exact.

This is not a restart of the wide LR search from earlier. The hypothesis is narrower: once explicit regularization is present and the activation is smoother, a slightly smaller step can sometimes preserve more of the generalization gain without materially hurting progress inside the same 300-second budget. If it fails, that still helps tighten the basin around the current best setting.

## Slightly higher learning rate inside the AdamW baseline
After testing the lower side of the local learning-rate basin, the complementary question is whether a small increase above `4e-4` is still safe in the new regularized regime. Earlier large-step experiments were too aggressive, but those were done before the current `SiLU + AdamW + wd=1e-4` setup existed, so the stability boundary may have shifted.

This is again a local identification step rather than a blind escalation. The goal is to determine whether the best current setting is a genuine local optimum or just the middle point between one bad low-side run and an untested high side. `4.5e-4` is close enough to be informative without turning into a qualitatively different optimizer regime.

## Smaller `digit1h` head for a simpler and faster baseline
The current kept model still uses a fairly large head on top of the shared three-tower representation. At this point the optimizer search has mostly converged, so a good next question is whether some of that head capacity is unnecessary and whether trimming it can buy either better generalization or more useful updates per 300-second run.

This matches the repo’s simplicity criterion well. If a smaller head reaches the same or better eval loss, that is a clear win because it removes parameters and may improve throughput at the same time. I am only shrinking the head, not the shared tower, so the experiment stays local and keeps the learned chunk decomposition intact.

## Light dropout on the concatenated tower representation
There is already a dropout module sitting unused in the `digit1h` tower model, which makes it a very cheap regularization probe. Since the best current configuration uses AdamW and a fairly expressive head, applying a small amount of dropout right after concatenating the three tower outputs is a direct way to test whether the model is relying too heavily on brittle co-adaptations between chunk features.

The theoretical motivation is standard ensemble-style regularization: dropout perturbs intermediate features during training and can improve generalization if the head is over-specializing to particular activation patterns. Because the hook already exists in the code, this is also a nice low-complexity test under the repo’s simplicity criterion.

## Concatenate raw `digit1h` input with the tower representation
The three shared towers are good at building chunk-level features, but they also compress the original one-hot digits into a much smaller representation before the head sees them. A plausible failure mode is that some useful fine-grained digit information is being thrown away, especially for cross-chunk effects where the head might benefit from both an abstract summary and the exact raw digits.

This is a simple skip-connection style idea: let the head consume the concatenation of the tower outputs and the original `digit1h` input. In theory this makes optimization easier because the head no longer depends entirely on the towers to preserve low-level detail, while the towers can still learn reusable chunk abstractions. It adds some width to the head input, but the change is conceptually clean and reversible.

## Final-bias initialization closer to the label mean
The model initializes its output-layer bias to `96.0`, but the observed label mean is closer to `92-93` on both train and eval. Because the objective is absolute error on a scalar target and runs are time-limited, the starting offset can matter more than it would in a fully converged regime: a better prior mean may let the network spend more of its 300-second budget learning structure instead of first correcting a global bias term.

This is a deliberately simple initialization experiment. It does not change model capacity, optimizer dynamics, or feature processing; it only changes the prior prediction the network starts from. If it helps even slightly, it would be one of the cleanest wins so far because it improves the baseline without adding any architectural complexity.

## Wider per-chunk tower output bottleneck
The current architecture compresses each 3-digit chunk into only 30 learned features before the head combines the three chunks. That may simply be too tight a bottleneck: the shared tower could be learning useful chunk structure but discarding detail that would help the head reason about composition across millions, thousands, and the final chunk.

This is a targeted capacity increase rather than a broad model inflation. I am not widening the entire head or all hidden layers; I am only increasing the dimensionality of the tower output that the head consumes. If the current bottleneck is too lossy, this should help without the overfitting risk of exposing the full raw input directly, as happened with the skip-connection trial.

## Learned scalar chunk summaries alongside the tower embeddings
The task looks partly additive across 3-digit chunks, but the previous hand-designed structural features were too brittle and the raw-input skip was too unconstrained. A middle ground is to let the shared tower produce its usual embedding while also learning a single scalar summary for each chunk, then hand both the embeddings and the three scalar summaries to the final head.

This adds a very small amount of inductive bias without hard-coding the composition rule. If the network really benefits from thinking in terms of per-chunk contributions, these learned scalar summaries should make that easier; if not, the head can mostly ignore them. It is a cheap experiment in structured representation rather than a full architectural rewrite.

## LayerNorm on the concatenated tower features
The current best model relies on a fairly deep head operating on the concatenation of three tower outputs. A small normalization layer at that interface is a cheap way to test whether some of the remaining instability is coming from feature-scale drift across chunks rather than from the optimizer itself.

This is a minimal intervention: the towers, head widths, optimizer, and training loop all stay the same. If LayerNorm helps, it would point toward representation conditioning as the limiting factor; if it hurts, that suggests the current towers are already producing a well-scaled interface and extra normalization just injects noise.

## Shallower final head on top of the shared towers
The current best architecture uses a fairly deep head after the three chunk towers, which may be more expressiveness than this scalar regression actually needs. Since the towers already do substantial representation learning, it is plausible that a shorter head would optimize more cleanly within the fixed 300-second budget and regularize away some unnecessary nonlinear mixing.

This is a simplicity-first experiment. Unlike widening or adding skip paths, reducing head depth cuts parameters and compute. If the task’s hard part is mostly in the chunk towers and not in the final composition network, a shallower head could be a genuine win rather than just a neutral simplification.

## Slightly shallower shared chunk tower
The last simplification test on the head did not beat the best model, but it did show that simpler variants can still get reasonably close while running a bit faster. That makes the shared tower the next natural place to probe for excess depth: perhaps the tower is doing more nonlinear processing than the 3-digit subproblem actually requires.

This is still a local simplification, not a wholesale redesign. I am trimming one internal stage from the shared tower while leaving the head, optimizer, and input representation unchanged. If the current tower is overbuilt, this could preserve most of the useful chunk abstraction while freeing budget for more parameter updates.

## Cosine learning-rate decay within the fixed time budget
Most of the optimizer experiments so far have only changed the constant base learning rate, but under a hard wall-clock cap there is a reasonable case for using a schedule instead of a fixed step size. A cosine decay keeps the early updates large enough to move quickly, then reduces the step size later when the model is already in a better basin and small refinements matter more.

This is a useful next axis because it changes optimization dynamics without changing the model at all. If the current baseline is spending too much of its late training time bouncing around with an overly large constant LR, a cheap scheduler could improve eval with very little code complexity.

## Gradient clipping for rare unstable updates
The current baseline occasionally shows noisy epoch-to-epoch jumps even when the overall trend is reasonable. One possible explanation is that a small number of unusually large updates are knocking the model around more than they should, especially with a fairly expressive shared tower and head under AdamW.

Gradient clipping is a simple way to test that hypothesis without changing the optimizer or architecture. If clipping helps, it suggests the issue is update stability rather than representation quality; if it does nothing or hurts, then the current noise is probably the useful kind of stochasticity rather than pathological spikes.

## Recheck the plain `digit` representation under the stronger optimizer regime
Most of the recent progress has come from tuning the `digit1h` model, but that does not mean the representation search itself is closed. The earlier branch history mostly compared input types before the current optimizer settings were found, so it is still possible that a simpler raw-digit representation becomes competitive once trained with `AdamW`, a smaller learning rate, and the full 300-second budget.

This is worth testing because it changes the inductive bias of the entire system with zero code complexity. If `digit` can get anywhere close, it would be much simpler than the chunked one-hot tower architecture; if it is still far worse, that at least confirms the current representation choice is doing real work and is not just an artifact of hyperparameter tuning.

## Separate towers for millions, thousands, and final chunk
The current `digit1h` model uses one shared tower for all three 3-digit chunks, which assumes those positions should be processed identically. That symmetry is attractive, but it may simply be wrong for this task: the million and thousand chunks always contribute suffix structure, while the final chunk has the special `"and"` behavior and directly controls the low-order wording in a different way.

Unsharing the towers is a bigger architectural move than the recent local tweaks, but the rationale is strong. If the current shared-tower constraint is hiding real position-specific structure, separate towers should let the model specialize its representations without resorting to brittle hand-coded features. It costs more parameters, but unlike the raw-input skip, the added capacity is aligned with a clear task asymmetry.

## Intermediate tower bottleneck width of 48
The previous bottleneck widening experiment to 64 improved over several other architectural changes but still did not beat the best run. That suggests the basic direction might be right while the exact capacity tradeoff is off: 30 may be a little too tight, but 64 may already be larger than the head can exploit efficiently within the time budget.

Testing 48 is a direct interpolation along that curve. If the bottleneck really is the constraint, a moderate increase should preserve most of the extra representational room while staying closer to the computational footprint and regularization behavior of the best current baseline.

## Moderate larger-batch retry at 1536
The earlier `2048` batch-size trial was too large, but that does not rule out a smaller move on the high side. Because the baseline is still optimizer-limited in a fixed 300-second window, there is a plausible throughput argument for trying a moderately larger batch that might increase examples processed without pushing all the way into the bad large-batch regime.

`1536` is a deliberately cautious retry. It preserves most of the baseline’s update scale while changing the compute/noise tradeoff less violently than `2048`. If it is still worse, that would strengthen the case that the useful batch-size region is centered close to the current `1024`.
