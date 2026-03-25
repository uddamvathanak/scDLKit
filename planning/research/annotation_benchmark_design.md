# Annotation Benchmark Design: Research and Rationale

## Research Sources

- [scGPT (Nature Methods 2024)](https://www.nature.com/articles/s41592-024-02201-0): Foundation model for single-cell multi-omics
- [scGPT Annotation Tutorial](https://scgpt.readthedocs.io/en/latest/tutorial_annotation.html): Official fine-tuning protocol
- [scGPT Nature Protocols](https://www.nature.com/articles/s41596-025-01220-1): End-to-end protocol for retinal cell annotation
- [PEFT for scLLMs (PMC10862733)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10862733/): Systematic PEFT comparison (LoRA, prefix, adapter)
- [scPEFT (Nature Machine Intelligence 2025)](https://www.nature.com/articles/s42256-025-01170-z): Parameter-efficient adapters for scLLMs
- [Zero-shot evaluation (Genome Biology 2025)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x): Limitations of foundation models
- [Atlas-level integration benchmark (Nature Methods)](https://www.nature.com/articles/s41592-021-01336-8): Benchmark design patterns

## Current State (Problems)

| Aspect | Before | After (implemented) |
|--------|--------|---------------------|
| Epochs | 2-3 per strategy | 10-15 with early stopping (matching scGPT protocol) |
| Early stopping patience | 3 | 5 |
| LR schedule | None (constant) | StepLR gamma=0.9 |
| Data size | max_cells=64 (CI profile) | Full dataset (no subsampling) |
| Metrics | accuracy, macro_f1, balanced_accuracy, AUROC | + weighted_f1, precision, recall, Cohen's kappa, MCC, per-class |
| Figures | Basic bar charts | Publication-quality with error bars, confidence bands, heatmaps, radar |
| Per-class analysis | None | Per-class F1 heatmap |
| Error bars | None (single mean) | Mean +/- std across 3 seeds |

## Benchmark Design (Based on Literature)

### Training Protocol

Following scGPT (Nature Methods 2024) and PEFT for scLLMs (PMC10862733):

- **Epochs**: 10-15 max (scGPT uses 10 for annotation, PEFT paper uses up to 100)
- **Early stopping**: patience=5, monitoring validation loss
- **LR schedule**: StepLR with gamma=0.9 per epoch (matching scGPT protocol)
- **Convergence**: Best model checkpoint retained based on validation loss
- **Batch size**: 64 (current, reasonable for these dataset sizes)
- **Data**: Full datasets without subsampling (PBMC: 700 cells, Pancreas: ~2700 cells)

Strategy-specific hyperparameters:

| Strategy | Epochs | LR | Notes |
|----------|--------|----|-------|
| head | 15 | 5e-3 | Only classifier head trainable |
| full_finetune | 10 | 5e-4 | Lower LR, fewer epochs to prevent catastrophic forgetting |
| lora | 15 | 2e-3 | rank=4, alpha=8.0 |
| adapter | 15 | 2e-3 | bottleneck_dim=64 |
| prefix_tuning | 15 | 2e-3 | prefix_length=20 |
| ia3 | 15 | 2e-3 | init_scale=1.0 |

### Metrics Suite

**Primary (reported in main text):**
- Macro F1 (standard across all papers)
- Weighted F1 (handles class imbalance)
- Accuracy
- Balanced accuracy

**Secondary (supplements and detailed analysis):**
- AUROC (one-vs-rest, macro)
- Cohen's kappa (agreement beyond chance)
- Matthews correlation coefficient (robust for imbalanced classes)
- Macro precision
- Macro recall

**Per-class:**
- Per-class F1, precision, recall (for heatmap visualization)
- Confusion matrix

**Efficiency:**
- Trainable parameter count / fraction
- Wall-clock runtime
- Peak memory
- Checkpoint size

### Figure Inventory (Publication-Ready)

**Main text figures:**

1. **Performance comparison** (Fig 2-style):
   - Grouped bar chart with error bars (mean +/- std across 3 seeds)
   - One panel per dataset (PBMC, Pancreas)
   - Shows macro_f1, weighted_f1, balanced_accuracy side by side
   - Strategies ordered from baseline to most complex PEFT
   - Color palette: Nature-style muted colors

2. **Label efficiency curves** (Fig 3-style):
   - Line plot with shaded confidence bands (std across seeds)
   - X-axis: label fraction (1%, 5%, 10%, 100%)
   - Y-axis: macro F1
   - One line per strategy
   - Story: PEFT methods degrade less with fewer labels than full FT

3. **Cross-study robustness** (Fig 4-style):
   - Heatmap: rows=strategies, columns=cross-study folds
   - Color intensity = macro F1
   - Shows which methods are robust to distribution shift
   - Annotated with values

4. **Efficiency-performance Pareto** (Fig 5-style):
   - Scatter: x=trainable parameters (log scale), y=macro F1
   - Each point labeled with strategy name
   - Pareto frontier line drawn
   - Bubble size = runtime
   - Story: PEFT achieves comparable performance with <10% parameters

**Supplementary figures:**
- Per-class F1 heatmaps per dataset
- Confusion matrices for best strategies
- UMAP embeddings
- Full metrics tables as CSV

### Narrative Structure

The benchmark tells this story:

1. **Baseline**: PCA + logistic regression and frozen probe show what the pretrained model can do without any adaptation
2. **Head-only**: Minimal adaptation already improves performance
3. **Full fine-tuning**: Gold standard but expensive (all parameters trainable)
4. **PEFT methods**: LoRA, adapter, prefix tuning, IA3 match or exceed full FT with fraction of parameters
5. **Stress tests**: Low-label and cross-study regimes reveal which methods are robust
6. **Pareto frontier**: Demonstrates the practical trade-off for deployment decisions

## Implementation Plan

1. Add comprehensive metrics to `classification_metrics()` in `src/scdlkit/evaluation/metrics.py`
2. Update epoch counts in `_foundation_annotation_profile()` in `scripts/run_quality_suite.py`
3. Update early stopping patience and add LR scheduler in benchmark script
4. Add LR scheduler support to `Trainer` class
5. Redesign all figure functions in `scripts/run_annotation_benchmark.py`
6. Add new figure types (radar chart, per-class heatmap)
7. Update report payload to include new metrics
8. Run full benchmark with `--resume` flag to avoid re-running PCA baseline
