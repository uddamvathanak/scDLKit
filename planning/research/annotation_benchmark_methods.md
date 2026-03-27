# Annotation Benchmark Methods

This note explains the methods used in the Milestone 1 annotation benchmark in
plain language.

The goal is to make it easy to reread the benchmark outputs later and remember:

- what each method actually does
- how we implement it in this repo
- whether the benchmark step is supervised, unsupervised, or uses a pretrained
  self-supervised backbone
- what is trained and what stays frozen
- why that method is included in the comparison

## Common benchmark setup

All methods are evaluated under the same benchmark regimes so the comparison is
fair.

### Full-label regime

- The dataset is split into train, validation, and test sets.
- The current benchmark code uses approximately:
  - `70%` train
  - `15%` validation
  - `15%` test
- Splits are stratified by label when possible, so class balance is preserved.
- Models train only on the training split.
- The validation split is used for model selection and early stopping when the
  method supports training.
- Final reported metrics come from the held-out test split.

### Low-label regime

- Start from the same full-label split.
- Keep validation and test the same.
- Subsample the training labels to smaller fractions such as `1%`, `5%`, and
  `10%`.
- This tests how well each method works when labeled data is scarce.

### Cross-study regime

- The held-out test set is defined by batch or technology rather than by random
  cells.
- For pancreas, specific batch families are held out entirely as the test fold.
- The remaining data is split into train and validation.
- This is a harder test because the model must generalize across study or
  technology shift, not just across random train/test partitions.

## How the benchmark is run in this repo

For the scGPT methods, the benchmark is not feeding raw text-like tokens into a
language model. It converts each cell into a structured sequence that scGPT can
process.

The pipeline is:

1. Start from a single cell's expression vector.
2. Keep only genes that match the scGPT checkpoint vocabulary.
3. For each cell, keep the non-zero genes and their expression values.
4. Insert a `<cls>` token at the front of the sequence.
5. Bin the expression values into discrete levels expected by scGPT.
6. Pad or truncate the sequence to the model's max length.
7. Feed three aligned tensors into scGPT:
   - `gene_ids`
   - `values`
   - `padding_mask`
8. Take the pooled cell representation from the `<cls>` position.
9. Use that representation either:
   - as a frozen embedding for a separate classifier, or
   - as the input to a trainable annotation head

So the benchmark is still a standard supervised annotation benchmark in the
Milestone 1 sense:

- train on labeled training cells
- use validation for trainable strategies
- report final metrics on held-out test cells

The main exception is that some methods use an unsupervised or pretrained stage
inside the pipeline:

- PCA is an unsupervised dimensionality reduction step followed by supervised
  classification.
- Frozen scGPT uses a pretrained self-supervised backbone followed by a
  supervised linear probe.
- The tuned scGPT methods are supervised adaptation on labeled cells.

## How single-cell inputs differ from text or images

The word "token" can be misleading if you come from NLP.

In this benchmark:

- a token is not a word
- a token usually corresponds to a gene identity
- the accompanying value is not a word embedding position or pixel intensity
  grid location
- the value is the cell's expression level for that gene

For one cell, you can think of the input as:

- gene identities present in that cell
- expression strengths for those genes
- one special `<cls>` token at the front for pooling the cell representation

A rough toy example is:

- cell has non-zero genes `CD3D`, `LTB`, `IL7R`
- matched gene IDs might look like `[104, 582, 991]`
- expression values might look like `[3.2, 1.4, 4.8]`
- after adding the class token, the model sees something like:
  - `gene_ids = [<cls>, 104, 582, 991, ...pad...]`
  - `values = [0, 3.2, 1.4, 4.8, ...pad...]`

That is why scGPT in this repo uses both gene embeddings and value embeddings,
then adds them before the transformer stack.

## Learning type labels used below

To make the benchmark easier to interpret, use these meanings:

- `unsupervised + supervised`: an unsupervised feature step plus a supervised
  classifier on labels from the training split
- `pretrained self-supervised + supervised probe`: a backbone pretrained
  elsewhere without labels, then frozen here, with a supervised classifier on
  top
- `supervised fine-tuning`: labeled training cells directly update some
  trainable part of the annotation model in this benchmark

Important scope note:

- Milestone 1 does not benchmark a self-supervised adaptation path inside the
  repo yet
- the self-supervised part is the scGPT checkpoint pretraining that happened
  before our benchmark
- the actual annotation benchmark in this repo is still label-supervised at
  evaluation time

## Why multiple methods are benchmarked

The benchmark is not only asking "which method gets the highest score?"

It is also asking:

- How strong is a simple classical baseline?
- How much value does pretrained scGPT provide without any tuning?
- When do we need tuning?
- Can parameter-efficient tuning match full fine-tuning?
- Which methods remain competitive under low-label and cross-study stress?

That is why the matrix includes both classical and foundation-model methods.

## Method-by-method explanation

### 1. PCA + logistic regression

What it is:

- `PCA` reduces the gene-expression matrix to a smaller latent representation.
- `LogisticRegression` is then trained on the PCA features from the training
  split.
- The fitted PCA transform is applied to the held-out test split, and the
  classifier predicts labels there.

Learning type:

- `unsupervised + supervised`

What is trained:

- PCA is fit on the training expression matrix.
- Logistic regression is trained on the training labels.
- There is no deep neural network and no foundation model here.

How we do it in this repo:

- Take the training expression matrix.
- Fit PCA on training cells only.
- Transform both train and test cells into PCA latent coordinates.
- Train logistic regression on the train latents and train labels.
- Score on the held-out test labels.

Why it is included:

- It is the main classical baseline.
- It tells us whether a simple linear pipeline already solves most of the task.
- If PCA + logistic regression performs close to a fancy method, the extra
  complexity may not be justified.
- It keeps the benchmark honest by comparing scGPT against a cheap standard
  reference, not only against other scGPT variants.

How to interpret it:

- Strong PCA performance means the label signal is already easy to separate in a
  simple low-dimensional linear space.
- Weak PCA performance creates a stronger case for pretrained representations or
  adaptation.

### 2. Frozen scGPT probe

What it is:

- Use pretrained scGPT as a fixed feature extractor.
- scGPT produces embeddings for the training and test cells.
- A separate logistic regression classifier is trained on the training
  embeddings.
- Test predictions come from the held-out test embeddings.

Learning type:

- `pretrained self-supervised + supervised probe`

What is trained:

- The scGPT backbone is not updated.
- Only the external linear classifier on top of the embeddings is trained.

How we do it in this repo:

- Convert each cell into scGPT-ready `gene_ids`, `values`, and `padding_mask`.
- Run the frozen scGPT backbone on the train and test cells.
- Use the pooled latent embedding from scGPT.
- Train logistic regression on the training embeddings and labels.
- Evaluate the logistic regression classifier on held-out test embeddings.

Why it is included:

- It measures the value of the pretrained representation alone.
- It answers: "If we do not fine-tune scGPT at all, how much annotation signal
  is already present in the embedding?"
- This is the cleanest baseline for separating representation quality from
  tuning quality.

How to interpret it:

- If frozen scGPT beats PCA clearly, the pretrained model is adding useful
  biological prior.
- If frozen scGPT is already close to tuned scGPT, tuning may offer only modest
  practical benefit on that dataset.

### 3. scGPT head-only tuning

What it is:

- Keep the scGPT backbone frozen.
- Attach a trainable classifier head.
- Train only that head using labeled training cells.

Learning type:

- `supervised fine-tuning`

What is trained:

- Only the classifier head.
- The scGPT backbone remains frozen.

How we do it in this repo:

- Load the scGPT backbone.
- Freeze backbone parameters.
- Add a small classifier head on top of the pooled cell embedding.
- Train the head with cross-entropy on labeled training cells.
- Use validation for early stopping and report final metrics on the test split.

Why it is included:

- It is the cheapest true adaptation baseline.
- It tests whether the pretrained embedding is already good enough that only a
  small supervised readout layer is needed.
- It is more integrated than the frozen-probe logistic baseline because the head
  is part of the model path, but it still has a very small trainable footprint.

How to interpret it:

- If head-only tuning performs much better than the frozen probe, then a small
  amount of supervised adaptation is worthwhile.
- If it is already near full fine-tuning, then heavier tuning methods may not
  be worth the extra cost.

### 4. scGPT full fine-tuning

What it is:

- Train the full scGPT annotation model end to end on the labeled training set.
- Both the backbone and classifier head are updated.

Learning type:

- `supervised fine-tuning`

What is trained:

- Essentially the whole trainable model.

How we do it in this repo:

- Load the scGPT annotation model.
- Unfreeze the backbone.
- Train backbone plus classifier head on labeled training cells.
- Use the validation split for checkpoint selection and early stopping.
- Report the final metrics on the held-out test split.

Why it is included:

- It is the strongest and most expensive adaptation baseline.
- It serves as the reference point for the PEFT methods.
- Without full fine-tuning in the benchmark, it would be hard to judge whether
  LoRA, adapters, prefix tuning, or IA3 are actually giving good efficiency
  trade-offs.

How to interpret it:

- High performance here shows the best result available when cost is less of a
  concern.
- If PEFT methods match or nearly match full fine-tuning, that is a strong
  result because they use far fewer trainable parameters.

### 5. scGPT LoRA

What it is:

- `LoRA` adds low-rank trainable matrices to selected backbone layers instead of
  updating the full backbone weights directly.

Learning type:

- `supervised fine-tuning`

What is trained:

- The LoRA parameters and the classifier head.
- The original backbone weights stay frozen.

How we do it in this repo:

- Load the scGPT backbone.
- Freeze the original backbone weights.
- Inject LoRA modules into selected transformer/feed-forward projection layers.
- Train only the LoRA parameters plus the classifier head on labeled cells.
- Evaluate on the held-out test split.

Why it is included:

- LoRA is one of the most common PEFT methods in large-model adaptation.
- It is a strong practical baseline for "near full-tuning performance at much
  lower cost."
- It helps answer whether low-rank updates are enough for cell-type annotation.

How to interpret it:

- Strong LoRA performance means the task can be solved by relatively small,
  structured updates to the pretrained model.
- It is especially interesting when runtime and trainable parameter count are
  much lower than full fine-tuning.

### 6. scGPT adapters

What it is:

- Insert small bottleneck adapter modules into the network.
- Keep the original backbone mostly frozen.
- Train the adapter modules and the classifier head.

Learning type:

- `supervised fine-tuning`

What is trained:

- Adapter layers and the classifier head.
- The base backbone weights remain frozen.

How we do it in this repo:

- Freeze the base scGPT backbone.
- Insert bottleneck adapter blocks into the model.
- Train only those adapter blocks and the classifier head.
- Keep the original pretrained weights fixed.

Why it is included:

- Adapters are a standard PEFT family and are especially natural when you want
  small task-specific modules on top of a shared frozen backbone.
- They provide a different PEFT trade-off than LoRA: additive bottleneck blocks
  instead of low-rank weight updates.

How to interpret it:

- Good adapter performance suggests that task-specific residual capacity is
  enough without touching the whole pretrained model.
- This is useful if you want to keep separate compact task modules for different
  datasets or annotation settings.

### 7. scGPT prefix tuning

What it is:

- Learn a small trainable prefix or prompt-like set of vectors that steer the
  transformer attention behavior.
- The main backbone weights stay frozen.

Learning type:

- `supervised fine-tuning`

What is trained:

- Prefix parameters and the classifier head.
- The original backbone remains frozen.

How we do it in this repo:

- Replace the normal transformer encoder with a prefix-tuned wrapper.
- For each transformer layer, create a small trainable table of prefix vectors.
- At each layer, concatenate those prefix vectors in front of the real cell
  token sequence.
- Run attention over `prefix + real tokens`.
- After the layer finishes, drop the prefix positions and keep only the updated
  real-token outputs for the next layer.

So yes: in practice it is very close to "a trainable vector as part of the
input", but with an important detail:

- in this repo it is not just one vector
- it is a small set of trainable vectors
- and they are applied again at every transformer layer, not only once at the
  raw input layer
- they are not real genes; they are learned control vectors

Small toy example:

- suppose a cell sequence after preprocessing is:
  - `[<cls>, CD3D, LTB, IL7R]`
- suppose prefix length is `2`
- at one transformer layer, the actual layer input becomes conceptually:
  - `[P1, P2, <cls>, CD3D, LTB, IL7R]`
- `P1` and `P2` are learned vectors, not gene tokens from the data
- attention is computed over that longer sequence
- after the layer finishes, the prefix positions are removed, so the next layer
  again keeps only the updated outputs for:
  - `[<cls>, CD3D, LTB, IL7R]`
- then that next layer gets its own prefix vectors inserted again

Why people like prefix tuning:

- it gives the frozen model a trainable "context" that nudges the attention
  computation
- it can adapt behavior without rewriting the original backbone weights
- it usually has a very small trainable parameter count

What prefix tuning is not:

- it is not adding fake gene-expression rows to the dataset
- it is not changing the biological input itself
- it is not the same as full fine-tuning
- it is better thought of as learned control tokens injected inside the model
  computation

Why it is included:

- Prefix tuning is another established PEFT method with a very small trainable
  footprint.
- It tests whether the model can be effectively redirected by learned prompts
  rather than direct weight updates.

How to interpret it:

- Strong prefix-tuning results mean the annotation task can be solved by
  steering the pretrained model rather than rewriting it.
- Weak results would suggest the task needs deeper adaptation than prompt-style
  control provides.

### 8. scGPT IA3

What it is:

- `IA3` applies trainable multiplicative scaling to selected internal modules.
- It is one of the most lightweight PEFT approaches in the benchmark.

Learning type:

- `supervised fine-tuning`

What is trained:

- Small learned scaling parameters and the classifier head.
- The main backbone weights stay frozen.

How we do it in this repo:

- Freeze the pretrained backbone.
- Insert trainable multiplicative scaling factors into selected internal
  modules.
- Train those small scaling parameters and the classifier head on the labeled
  training cells.

Why it is included:

- IA3 is useful as an ultra-light PEFT baseline.
- It tests whether extremely small intervention points are enough to adapt
  scGPT to annotation.

How to interpret it:

- If IA3 stays competitive, that is strong evidence that annotation does not
  need large trainable updates.
- If it falls behind LoRA or adapters, that suggests the task benefits from a
  bit more adaptation capacity.

## Practical reading order for the results

When you look at the benchmark outputs, a good order is:

1. `PCA + logistic regression`
   Reason: sets the classical baseline.
2. `Frozen scGPT probe`
   Reason: measures pretrained representation value without tuning.
3. `Head-only tuning`
   Reason: measures the cheapest integrated adaptation.
4. `Full fine-tuning`
   Reason: serves as the strongest expensive reference.
5. `LoRA`, `adapters`, `prefix tuning`, `IA3`
   Reason: these are the efficiency-oriented PEFT methods you compare against
   full fine-tuning.

That progression makes the benchmark story easier to read:

- classical baseline
- pretrained baseline
- minimal adaptation
- maximal adaptation
- efficient adaptation alternatives

## Randomness and reproducibility

### 5-fold stratified cross-validation

The benchmark uses **5-fold stratified cross-validation** as the standard
evaluation protocol. This is the gold-standard approach: every cell appears in
the held-out test set exactly once across the 5 folds, giving an unbiased
performance estimate.

Implementation:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` from
  scikit-learn partitions the dataset into 5 folds, preserving class balance.
- For each fold, the held-out fold (~20% of cells) is the test set.
- The remaining ~80% is further split into train (~85% of remaining = ~68%
  overall) and validation (~15% of remaining = ~12% overall).
- The validation set is used for early stopping and model selection in
  trainable strategies.

### What the base seed controls

A single base seed (default **42**) governs all sources of randomness:

| Source of randomness | How the seed is used |
|---|---|
| **K-fold partitioning** | `StratifiedKFold(random_state=42)` produces 5 deterministic folds. |
| **Train / validation split** | Within each fold, the remaining cells are split into train/val using `random_state = base_seed + fold_index`. |
| **Low-label subsampling** | When the regime is `low_label`, `base_seed + fold_index` controls which subset of training cells is retained. |
| **Model weight initialization** | For trainable strategies, `base_seed + fold_index` seeds PyTorch's random state for the classifier head and PEFT modules. |
| **Training stochasticity** | Mini-batch ordering, dropout masks, and other stochastic operations use the per-fold seed. |
| **PCA + logistic regression** | `random_state = base_seed + fold_index` for scikit-learn's PCA and LogisticRegression. |

### Cross-study regime

The cross-study regime uses **leave-one-technology-out** evaluation, which is
itself a form of cross-validation:

- Each technology family (plate-like, CEL-Seq family, droplet family) is held
  out as the test set in turn.
- The remaining cells are split into train/val with a fixed seed.
- Variance comes from the 3 held-out technology folds, not from repeated seeds.

### How standard deviation is computed

For each unique (strategy, dataset, regime) group:

1. Collect the metric value (e.g. Macro F1) from each of the 5 folds.
2. Compute `mean = (v₁ + v₂ + v₃ + v₄ + v₅) / 5`.
3. Compute `std = standard deviation across the 5 values`.

In the figures:

- **Performance bars** (panel a): error bars show **mean ± 1 std** across the
  5 folds.
- **Low-label curves** (panel b): shaded bands show **mean ± 1 std** at each
  label fraction.
- **Cross-study heatmap** (panel c): each cell shows the Macro F1 for that
  (strategy, held-out technology) pair (one value per fold, no aggregation).
- **Pareto scatter** (panel d): each point is positioned at the **mean** Macro
  F1 and **mean** trainable parameter count across 5 folds.
- **Radar chart** (panel e): each axis value is the **mean** across 5 folds
  (and both datasets).
- **Per-class F1 heatmap** (panel f): one heatmap per dataset, each using the
  **best fold** (by Macro F1) for each strategy.

### Total run count

- **Full-label**: 2 datasets × 5 folds × 8 strategies = **80 runs**
- **Low-label**: 2 datasets × 5 folds × 3 fractions × 8 strategies = **240 runs**
- **Cross-study**: 1 dataset × 3 technology folds × 8 strategies = **24 runs**
- **Total**: **344 runs**

### Manuscript language template

When writing the methods section, the following phrasing captures the design:

> Each strategy was evaluated using 5-fold stratified cross-validation
> (StratifiedKFold, random_state = 42). In each fold, the held-out partition
> (~20% of cells) served as the test set, and the remaining cells were split
> into training (~68%) and validation (~12%) sets. Training used early stopping
> on validation loss (patience = 5, max 10-15 epochs) with StepLR scheduling
> (gamma = 0.9). Reported metrics are the mean ± standard deviation across the
> 5 folds, computed on the held-out test set. For cross-study generalization,
> we used leave-one-technology-out evaluation.

### Why this is the standard for all milestones

This 5-fold stratified CV design is the benchmark standard for all milestones
(annotation, spatial, integration, perturbation). Using the same evaluation
protocol across milestones allows a single, consistent methods section in the
manuscript.

## Short summary

- Yes, the benchmark uses training data and held-out testing.
- In the normal full-label benchmark, each method is trained on the training
  split and reported on the held-out test split.
- The validation split is used for trainable methods such as head-only, full
  fine-tuning, LoRA, adapters, prefix tuning, and IA3.
- The PCA and frozen-probe baselines are included so the scGPT tuning methods
  are compared against meaningful simpler references.
- Prefix tuning in this repo is implemented as trainable per-layer prefix
  vectors concatenated before the real token sequence at each transformer layer.
