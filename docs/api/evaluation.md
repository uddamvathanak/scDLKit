# Evaluation and outputs

Use the evaluation helpers when you want consistent metrics, benchmark tables, and exported reports.

This section is where the public output contract becomes explicit:

- embeddings come back as `latent`
- reconstruction-capable models can expose `reconstruction`
- classification models expose `logits`
- reports can be exported to Markdown and CSV

```{eval-rst}
.. autofunction:: scdlkit.compare_models
```

```{eval-rst}
.. autofunction:: scdlkit.evaluation.evaluate_predictions
```

```{eval-rst}
.. autofunction:: scdlkit.evaluation.save_markdown_report
```

```{eval-rst}
.. autofunction:: scdlkit.evaluation.save_metrics_table
```
