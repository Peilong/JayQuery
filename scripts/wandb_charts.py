"""Export every logged metric from a single W&B run into one CSV.

Unlike a keyed scan_history() call, this makes no assumption about which
metrics exist or whether they were all logged on the same step.
"""

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENTITY = "jayquery"
PROJECT = "jayquery-vfl"
RUN_ID = "98wy6oor"  # the short id in the run URL (.../runs/<RUN_ID>), not the display name
OUTPUT_FILENAME = "wandb_run_charts.csv"

# Wall-clock columns W&B adds automatically. Keep them or drop them here.
KEEP_INTERNAL = ["_runtime", "_timestamp"]

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
print(f"Run: {run.name}  (id={run.id}, state={run.state})")

# scan_history() with no `keys` streams the full history for every metric.
# Passing `keys=[...]` would return only rows where ALL those keys are present,
# which drops any metric logged on a different cadence.
rows = list(run.scan_history())

if not rows:
    raise SystemExit("This run has no logged history.")

df = pd.DataFrame(rows)
print(f"Pulled {len(df)} raw history rows, {len(df.columns)} columns.")

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
if "_step" not in df.columns:
    raise SystemExit("No _step column in history; nothing to align on.")


def is_scalar_column(series: pd.Series) -> bool:
    """True if the column's values are plain numbers (CSV-representable)."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    return pd.api.types.is_number(non_null.iloc[0])


metric_cols = []
skipped_cols = []
for col in df.columns:
    if col == "_step" or col in KEEP_INTERNAL:
        continue
    if col.startswith("_"):  # _wandb and other internals
        continue
    (metric_cols if is_scalar_column(df[col]) else skipped_cols).append(col)

if skipped_cols:
    print(f"Skipped {len(skipped_cols)} non-scalar column(s) "
          f"(images/tables/plots can't go in a CSV): {', '.join(sorted(skipped_cols))}")

internal_cols = [c for c in KEEP_INTERNAL if c in df.columns]
df = df[["_step"] + internal_cols + sorted(metric_cols)]

# Metrics logged at different points in the loop land on separate, mostly-empty
# rows sharing a step. .first() ignores NaN, so this merges them into one row
# per step without overwriting real values.
df = df.groupby("_step", as_index=False).first().sort_values("_step")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_FILENAME, index=False)
print(f"\nWrote {len(df)} steps x {len(metric_cols)} metrics to '{OUTPUT_FILENAME}'")
print("Metrics exported:")
for col in sorted(metric_cols):
    print(f"  - {col}")