#!/usr/bin/env bash
# Reproduce every JayQuery paper result, and emit a verifiable, human-readable report.
#
# Rewritten 2026-07-30 around scripts/run_matrix.py (the hardened batch runner) plus the
# privacy audit and report generator. The previous version stubbed out experiments 2-5
# because run_deployment.py could not take a --run-config; run_matrix.py handles both run
# modes, so nothing is stubbed any more.
#
# TWO MODES
#   default   reuse artifacts that already exist; only run what is missing, then rebuild
#             the report. Minutes. Use while iterating.
#   --fresh   ignore every existing artifact and re-run everything end to end, so the
#             whole report comes from ONE coherent execution. Hours. Use when you are
#             confident in the configuration and want the numbers you will publish.
#
# Prerequisites (once — see README):
#   pip install -r requirements.txt && pip install -e .
#   bash scripts/download_oulad.sh
#
# Usage:
#   bash scripts/reproduce_all.sh                      # fill gaps, rebuild results/
#   bash scripts/reproduce_all.sh --fresh              # full rebuild for the paper
#   bash scripts/reproduce_all.sh --stages publish     # just rebuild results/ (seconds)
#   bash scripts/reproduce_all.sh --stages audit,publish --wandb
#   bash scripts/reproduce_all.sh --fresh --seeds "42 43 44" --dry-run
#
# EVERYTHING lands in results/ — one gitignored folder. Open results/README.md.

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
SEEDS="42 43 44 45 46"        # 5 seeds: DP arms have sd ~0.014, so 3 is marginal (§K)
ROUNDS=200                    # §K3 settled shared horizon
MODE="sim"                    # sim for sweeps; the deploy stage covers deployment
DEPLOY_SEEDS="42"             # deployment is ~27 min/run, so the headline arms use 1 seed
# Transmissions per student an averaging adversary sees: round(batch-sample-rate * rounds).
# 1 would model an artificially weak single-shot attacker.
ATTACK_OBS=10
PRIMARY="dedup_first"
VARIANTS="dedup_first dedup_best dedup_worst"
WANDB_GROUP="paper-v1"
ALL_STAGES="env,data,psi,baselines,matrix,deploy,eps,clip,audit,attack,publish"
STAGES="$ALL_STAGES"
FRESH=0
DRY_RUN=0
USE_WANDB=0

# Bare `python` is not on PATH in this project's conda/micromamba setup.
PY="${JQ_PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then PY="python"
  else PY="micromamba run -n jayquery python"; fi
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --fresh)    FRESH=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --wandb)    USE_WANDB=1; shift ;;
    --stages)   STAGES="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --deploy-seeds) DEPLOY_SEEDS="$2"; shift 2 ;;
    --attack-obs)   ATTACK_OBS="$2"; shift 2 ;;
    --rounds)   ROUNDS="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --group)    WANDB_GROUP="$2"; shift 2 ;;
    -h|--help)  sed -n '1,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (stages: $ALL_STAGES)" >&2; exit 2 ;;
  esac
done

[ -f pyproject.toml ] || { echo "Run from the repo root." >&2; exit 2; }

# ── helpers ──────────────────────────────────────────────────────────────────
stage_enabled () { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

say () { printf '\n\033[1m=== [%s] %s\033[0m\n' "$1" "$2"; }

# run <description> <command...>
run () {
  local desc="$1"; shift
  if [ "$DRY_RUN" = 1 ]; then printf '  DRY-RUN: %s\n' "$*"; return 0; fi
  printf '  -> %s\n' "$desc"
  "$@"
}

# have <artifact-path> — true when the artifact exists AND we are not in --fresh mode.
# This is the single place the two modes differ.
have () { [ "$FRESH" = 0 ] && [ -e "$1" ]; }

skip () { printf '  SKIP (exists: %s) — use --fresh to rebuild\n' "$1"; }

# run_matrix resumes completed jobs in reuse mode, and starts clean in fresh mode.
resume_flag () { [ "$FRESH" = 1 ] && echo "" || echo "--resume"; }

wandb_flags () {
  [ "$USE_WANDB" = 1 ] && echo "--wandb --wandb-group $WANDB_GROUP" || echo ""
}

START=$(date +%s)
echo "=== JayQuery: Reproducing All Experiments ==="
echo "Started:  $(date)"
echo "Mode:     $([ "$FRESH" = 1 ] && echo 'FRESH (full rebuild)' || echo 'reuse existing artifacts')"
echo "Stages:   $STAGES"
echo "Seeds:    $SEEDS   Rounds: $ROUNDS   Run mode: $MODE   Deploy seeds: $DEPLOY_SEEDS"
echo "Python:   $PY"

# ── env ──────────────────────────────────────────────────────────────────────
if stage_enabled env; then
  say env "Verifying the environment"
  run "import check" $PY -c "
import torch, flwr, opacus, autodp, wandb
from autodp.mechanism_zoo import GaussianMechanism
print(f'  torch {torch.__version__} | flwr {flwr.__version__} | '
      f'opacus {opacus.__version__} | wandb {wandb.__version__}')
"
  run "test suite" $PY -m pytest tests/ -q
fi

# ── data ─────────────────────────────────────────────────────────────────────
if stage_enabled data; then
  say data "Preparing dataset variants"
  if have "data/processed/$PRIMARY/manifest.json"; then
    skip "data/processed/$PRIMARY/manifest.json"
  elif [ "$FRESH" = 1 ]; then
    run "prepare_variants (forced)" $PY scripts/prepare_variants.py --force
  else
    run "prepare_variants" $PY scripts/prepare_variants.py
  fi
fi

# ── psi ──────────────────────────────────────────────────────────────────────
# Validates the architecture's alignment claim: the real ECDH protocol must reproduce the
# intersection prepare_silos.py computed directly (§G).
if stage_enabled psi; then
  say psi "Verifying PSI reproduces the pipeline intersection"
  run "verify_psi --all" $PY scripts/verify_psi.py --all
fi

# ── baselines ────────────────────────────────────────────────────────────────
# The comparison point for the whole paper: under the feasibility framing, VFL vs the
# architecture-matched centralized MLP IS the headline utility claim, so it runs across
# every seed rather than once (docs/action_plan.md §R6).
if stage_enabled baselines; then
  say baselines "Centralized + isolated-silo baselines ($(echo $SEEDS | wc -w) seeds)"
  for seed in $SEEDS; do
    if have "experiments/baselines_s${seed}.json"; then
      skip "experiments/baselines_s${seed}.json"
    else
      run "baselines seed $seed" $PY -m baselines.run_baselines \
        --config experiments/configs/exp01_centralized_baselines.yaml --run all \
        --processed-dir "data/processed/$PRIMARY" --seed "$seed" \
        --out "experiments/baselines_s${seed}.json"
    fi
  done
fi

# ── deploy: the headline arms in REAL federation mode ────────────────────────
# Simulation and deployment were measured as equivalent (§L5), but headline numbers come
# from deployment for provenance — it is the configuration a reader would actually
# reproduce. ~27 min/run, so this is one seed across the four arms by default.
if stage_enabled deploy; then
  say deploy "Headline arms in deployment mode (4 arms x $(echo $DEPLOY_SEEDS | wc -w) seed(s))"
  if have experiments/deploy_headline.json; then
    skip experiments/deploy_headline.json
  else
    run "run_matrix --jobs d3 --mode deployment" $PY scripts/run_matrix.py --jobs d3 \
      --variants "$PRIMARY" --seeds $DEPLOY_SEEDS --rounds "$ROUNDS" --mode deployment \
      --out experiments/deploy_headline.json $(resume_flag)
  fi
fi

# ── matrix (§D3): DP mode x dedup variant x seed ─────────────────────────────
if stage_enabled matrix; then
  say matrix "Utility matrix — 4 DP modes x $(echo $VARIANTS | wc -w) variants x $(echo $SEEDS | wc -w) seeds"
  if have experiments/d3_matrix.json; then
    skip experiments/d3_matrix.json
  else
    run "run_matrix --jobs d3" $PY scripts/run_matrix.py --jobs d3 \
      --variants $VARIANTS --seeds $SEEDS --rounds "$ROUNDS" --mode "$MODE" \
      --out experiments/d3_matrix.json $(resume_flag)
  fi
fi

# ── eps sweep: the privacy-utility curve (the core figure) ───────────────────
if stage_enabled eps; then
  say eps "Privacy-utility curve — cut-layer epsilon sweep"
  if have experiments/eps_sweep.json; then
    skip experiments/eps_sweep.json
  else
    # --max-norm-a/-b come from the clip calibration. Sweeping epsilon against a mis-set
    # clip maps the wrong curve entirely (see FINDINGS.md §2g).
    run "run_matrix --jobs eps" $PY scripts/run_matrix.py --jobs eps \
      --variants "$PRIMARY" --seeds $SEEDS --rounds "$ROUNDS" --mode "$MODE" \
      --wandb-group "eps-sweep" --max-norm-a 1.0 --max-norm-b 1.0 \
      --out experiments/eps_sweep.json $(resume_flag)
  fi
fi

# ── clip calibration: utility at IDENTICAL epsilon ───────────────────────────
# S does not appear in the epsilon calculation at all, so this measures pure utility
# spread along an epsilon-isocline (docs/privacy_presentation_plan.md §2.2).
if stage_enabled clip; then
  say clip "Clip calibration — utility spread at identical epsilon"
  if have experiments/clip_calibration.json; then
    skip experiments/clip_calibration.json
  else
    run "run_matrix --jobs clip" $PY scripts/run_matrix.py --jobs clip \
      --variants "$PRIMARY" --seeds $SEEDS --rounds "$ROUNDS" --mode "$MODE" \
      --out experiments/clip_calibration.json $(resume_flag)
  fi
fi

# ── audit: verify the privacy guarantee against each run's own evidence ──────
if stage_enabled audit; then
  say audit "Privacy audit (V1-V6)"
  run "verify_privacy" $PY scripts/verify_privacy.py \
    --variants $VARIANTS --out experiments/privacy_audit.json $(wandb_flags)
fi

# ── attack: empirical leakage vs epsilon, against the WIRE representation ────
if stage_enabled attack; then
  say attack "Attacks vs epsilon (wire-faithful, n=200)"
  if have experiments/attack_sweep.csv; then
    skip experiments/attack_sweep.csv
  else
    # One checkpoint per DP mode at the primary variant, seed 42. The no_dp arm is
    # included deliberately: it measures the ceiling instead of assuming one.
    TAGS=""
    for arm in no_dp server_only cut_only all_dp; do
      TAGS="$TAGS d3_${PRIMARY}_${arm}_s42"
    done
    run "run_attack --sweep" $PY scripts/run_attack.py \
      --processed-dir "data/processed/$PRIMARY" \
      --checkpoint-dir experiments/checkpoints \
      --sweep $TAGS --n-students 200 --observations "$ATTACK_OBS" \
      --results-csv experiments/attack_sweep.csv
  fi
fi

# ── publish: build results/ — the one folder a human opens ──────────────────
# Computes the claims (Welch deltas + CIs), draws the figures, writes every table as
# csv/tex/md, and re-detects the standing open issues from the artifacts so they cannot
# quietly vanish between sessions.
if stage_enabled publish; then
  say publish "Building results/"
  run "make_report" $PY scripts/make_report.py --out-dir results $(wandb_flags)
fi

ELAPSED=$(( $(date +%s) - START ))
echo
echo "=== Done: $(date)  (${ELAPSED}s) ==="
echo
echo "  ➜  START HERE:  results/README.md"
echo
echo "     results/claims.csv        every comparison, delta + 95% CI"
echo "     results/figures/          the paper's figures"
echo "     results/tables/           every table as csv / tex / md"
echo "     results/privacy_audit.md  the guarantee audit"
if [ "$USE_WANDB" = 1 ]; then
  echo "     wandb: https://wandb.ai/jayquery/jayquery-vfl (group: $WANDB_GROUP)"
  echo "            offline runs are uploaded with: wandb sync"
fi
