# Baseline Results — Week 3

## See more details on wandb
> https://wandb.ai/jayquery/baseline

| Model | Silo | Accuracy | AUC-ROC | Avg Precision | Brier Score | F1 (Binary) | F1 (Macro) | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | Merged | 0.968 | 0.9974 | 0.9982 | 0.0292 | 0.9728 | 0.9670 | 0.9795 | 0.9662 |
| Logistic Regression | Merged | 0.940 | 0.9784 | 0.9893 | 0.0488 | 0.9470 | 0.9389 | 0.9926 | 0.9054 |
| MLP (Full) | Merged | 0.896 | 0.9585 | 0.9755 | 0.0863 | 0.9078 | 0.8943 | 0.9552 | 0.8649 |
| Isolated Silo B | Silo B | 0.700 | 0.9096 | 0.9233 | 0.1924 | 0.6696 | 0.6974 | 0.9620 | 0.5135 |
| Isolated Silo A | Silo A | 0.544 | 0.5908 | 0.6920 | 0.2490 | 0.5169 | 0.5426 | 0.6932 | 0.4122 |

> Sorted by AUC-ROC descending. All runs: seed=42, batch_size=64, dropout=0.2, lr=0.001, epochs=50.