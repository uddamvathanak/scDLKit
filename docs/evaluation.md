# Evaluation

Built-in metrics include:

- Reconstruction: MSE, MAE, Pearson, Spearman
- Representation: silhouette, kNN label consistency, ARI, NMI, optional batch silhouette
- Classification: accuracy, macro F1, confusion matrix

`TaskRunner.evaluate()` returns a metrics dictionary. `TaskRunner.save_report()` exports Markdown and CSV artifacts.
