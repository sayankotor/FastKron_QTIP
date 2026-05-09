#!/usr/bin/env bash
#
# run_experiment_7b_true_accum.sh — full pipeline experiment for Llama-2-7B
# with TRUE-accumulation Fisher gradient collection.
#
# Phases:
#   1. BASELINE EVAL    — eval_ppl + eval_zeroshot on unsloth/llama-2-7b
#   2. FISHER           — run_chunked_fisher.sh with collect_fisher_weights_true_accum.py
#   3. QUANTIZE + HFize — quantize_finetune_llama.py + hfize_llama.py (K=4)
#   4. QUANTIZED EVAL   — same benchmarks on quantized_hf model
#   5. COMPARISON       — print summary table baseline vs quantized
#
# CLI: --skip-baseline | --skip-fisher | --skip-quantize | --force-baseline | --help
# Idempotency: per-phase done_phase_*.flag (and per-chunk in fisher wrapper).

set -euo pipefail
umask 022

# ---------- helpers ----------
stamp() { date +%H:%M:%S; }
log() { echo "[$(stamp)] [experiment] $*"; }

# ---------- locked experiment params ----------
MODEL_NAME="unsloth/llama-2-7b"
NUM_LAYERS=32
CHUNK_SIZE=16
DATASET_SIZE=9600
GRAD_ACCUM=128
MAX_LENGTH=3096
PER_DEVICE_BATCH_SIZE=1
ATTN_IMPL=sdpa
COLLECT_SCRIPT=collect_fisher_weights_true_accum.py

QUANT_K=4
QUANT_L=16
QUANT_V=2
QUANT_TLUT_BITS=9
QUANT_TD_X=16
QUANT_TD_Y=16

TASKS_LIST="arc_challenge,arc_easy,boolq,piqa,hellaswag,winogrande,gsm8k,ifeval"

# ---------- env-overridable ----------
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORK_DIR="${WORK_DIR:-/workspace-SR004.nfs2/chekalina/yaqa-quantization/kronfwsvd/grads_output/llama2_7b_true_accum_run}"

# ---------- repo paths ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PPL_PY="${REPO_ROOT}/eval/eval_ppl.py"
EVAL_LM_PY="${REPO_ROOT}/eval/eval_zeroshot.py"
QUANT_PY="${REPO_ROOT}/quantize_llama/quantize_finetune_llama.py"
HFIZE_PY="${REPO_ROOT}/quantize_llama/hfize_llama.py"
FISHER_WRAPPER="${REPO_ROOT}/run_chunked_fisher.sh"

# ---------- CLI ----------
SKIP_BASELINE=""
SKIP_FISHER=""
SKIP_QUANTIZE=""
FORCE_BASELINE=""

usage() {
    cat <<EOF
usage: $0 [--skip-baseline] [--skip-fisher] [--skip-quantize]
          [--force-baseline] [--help]

Full pipeline (Llama-2-7B, true accumulation Fisher, K=$QUANT_K):
  Phase 1: baseline eval (cached in baseline_eval.json)
  Phase 2: Fisher factors via $FISHER_WRAPPER
  Phase 3: quantize + hfize
  Phase 4: quantized eval (always recomputed)
  Phase 5: comparison summary

Locked params:
  MODEL_NAME=$MODEL_NAME  NUM_LAYERS=$NUM_LAYERS  CHUNK_SIZE=$CHUNK_SIZE
  DATASET_SIZE=$DATASET_SIZE  GRAD_ACCUM=$GRAD_ACCUM  MAX_LENGTH=$MAX_LENGTH
  PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE  ATTN_IMPL=$ATTN_IMPL
  COLLECT_SCRIPT=$COLLECT_SCRIPT
  Quantize: K=$QUANT_K L=$QUANT_L V=$QUANT_V tlut_bits=$QUANT_TLUT_BITS td=${QUANT_TD_X}x${QUANT_TD_Y}
  Tasks: $TASKS_LIST

Env-overridable:
  EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE
  WORK_DIR=$WORK_DIR

CLI flags:
  --skip-baseline   skip Phase 1 (require baseline_eval.json present, else fatal)
  --skip-fisher     skip Phase 2 (require factors/ populated, else fatal)
  --skip-quantize   skip Phase 3 (require quantized_hf/config.json, else fatal)
  --force-baseline  re-run Phase 1 ignoring cached baseline_eval.json
  --help            show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-baseline)   SKIP_BASELINE=1; shift ;;
        --skip-fisher)     SKIP_FISHER=1;   shift ;;
        --skip-quantize)   SKIP_QUANTIZE=1; shift ;;
        --force-baseline)  FORCE_BASELINE=1; shift ;;
        --help|-h)         usage; exit 0 ;;
        *)                 echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$SKIP_BASELINE" && -n "$FORCE_BASELINE" ]]; then
    echo "--skip-baseline and --force-baseline are mutually exclusive" >&2
    exit 2
fi

# ---------- paths ----------
LOGS_DIR="$WORK_DIR/logs"
FACTORS_DIR="$WORK_DIR/factors"
QUANTIZED_DIR="$WORK_DIR/quantized"
HF_DIR="$WORK_DIR/quantized_hf"
BASELINE_JSON="$WORK_DIR/baseline_eval.json"
QUANTIZED_JSON="$WORK_DIR/quantized_eval.json"

mkdir -p "$LOGS_DIR" "$FACTORS_DIR"

# ---------- ERR trap ----------
PHASE="pre-loop"
trap 'log "FAILED at phase ${PHASE}, line ${LINENO}, exit $?"; exit 1' ERR

# ---------- timing ----------
declare -A PHASE_TIME

# ---------- validate scripts exist ----------
for p in "$EVAL_PPL_PY" "$EVAL_LM_PY" "$QUANT_PY" "$HFIZE_PY" "$FISHER_WRAPPER" \
         "$REPO_ROOT/kronfwsvd/$COLLECT_SCRIPT" \
         "$REPO_ROOT/kronfwsvd/get_kron_factors_llama.py"; do
    [[ -f "$p" ]] || { log "FATAL: missing required script: $p"; exit 1; }
done

# =====================================================================
# Helper: merge eval JSONs into baseline_eval.json or quantized_eval.json
# =====================================================================
merge_eval_json() {
    local kind="$1"        # "baseline" or "quantized"
    local hf_path="$2"     # actual HF path used for eval
    local ppl_tmp="$3"
    local lm_tmp="$4"
    local out_path="$5"

    KIND="$kind" HF_PATH="$hf_path" PPL_TMP="$ppl_tmp" LM_TMP="$lm_tmp" \
        OUT_PATH="$out_path" MODEL_NAME_ENV="$MODEL_NAME" \
        python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

kind = os.environ["KIND"]
hf = os.environ["HF_PATH"]
ppl_path = Path(os.environ["PPL_TMP"])
lm_path = Path(os.environ["LM_TMP"])
out_path = os.environ["OUT_PATH"]
model = os.environ["MODEL_NAME_ENV"]

ppl = json.loads(ppl_path.read_text()) if ppl_path.exists() else {}
lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}

ppl_results = ppl.get("results", {})
lm_results = lm.get("results", {})

def primary(results, task, candidates):
    if task not in results:
        return None
    for k in candidates:
        if k in results[task]:
            v = results[task][k]
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
    return None

key_metrics = {
    "wikitext2_ppl": ppl_results.get("wikitext2_ppl"),
    "c4_ppl":        ppl_results.get("c4_ppl"),
    "arc_challenge": primary(lm_results, "arc_challenge", ["acc_norm,none", "acc,none"]),
    "arc_easy":      primary(lm_results, "arc_easy",      ["acc_norm,none", "acc,none"]),
    "boolq":         primary(lm_results, "boolq",         ["acc,none", "acc"]),
    "piqa":          primary(lm_results, "piqa",          ["acc_norm,none", "acc,none"]),
    "hellaswag":     primary(lm_results, "hellaswag",     ["acc_norm,none", "acc,none"]),
    "winogrande":    primary(lm_results, "winogrande",    ["acc_norm,none", "acc,none"]),
    "gsm8k_5shot":   primary(lm_results, "gsm8k",         ["exact_match,strict-match", "exact_match,flexible-extract"]),
    "ifeval_strict": primary(lm_results, "ifeval",        ["prompt_level_strict_acc,none", "prompt_level_strict_acc"]),
}

merged = {
    "_meta": {
        "model": model,
        "kind": kind,
        "hf_path_used": hf,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fewshot_note": (
            "All tasks use lm-eval YAML defaults (mostly 0-shot, gsm8k=5-shot, "
            "ifeval=0-shot); not comparable to OpenLLM Leaderboard which uses "
            "25-shot ARC, 10-shot HellaSwag, 5-shot Winogrande, etc."
        ),
    },
    "ppl": ppl,
    "lm_eval": lm,
    "key_metrics": key_metrics,
}

text = json.dumps(merged, indent=2, default=str)
tmp = out_path + ".tmp"
with open(tmp, "w") as f:
    f.write(text)
os.replace(tmp, out_path)
print(f"[merge] wrote {out_path}")
PY
}

# =====================================================================
# Helper: print metrics block (baseline or with-delta)
# =====================================================================
print_metrics_block() {
    local kind="$1"           # "BASELINE" or "QUANTIZED"
    local primary_json="$2"   # path to current-kind json
    local baseline_json="$3"  # path to baseline json (for deltas); empty if no delta

    KIND="$kind" PRIMARY="$primary_json" BASELINE="$baseline_json" \
        MODEL_NAME_ENV="$MODEL_NAME" python - <<'PY'
import json, os, time

kind = os.environ["KIND"]
primary_json = os.environ["PRIMARY"]
baseline_json = os.environ["BASELINE"]
model = os.environ["MODEL_NAME_ENV"]

with open(primary_json) as f:
    pm = json.load(f)["key_metrics"]
bm = None
if baseline_json:
    with open(baseline_json) as f:
        bm = json.load(f)["key_metrics"]

labels = ["wikitext2_ppl", "c4_ppl", "arc_challenge", "arc_easy",
          "boolq", "piqa", "hellaswag", "winogrande",
          "gsm8k_5shot", "ifeval_strict"]

def s():
    return time.strftime("%H:%M:%S")

# strip 'unsloth/' for shorter print
short_model = model.split("/")[-1]
print(f"[{s()}] [experiment] {kind} {short_model}:")
for label in labels:
    pv = pm.get(label)
    if isinstance(pv, (int, float)):
        val = f"{pv:.3f}"
    else:
        val = "N/A"
    if bm is None:
        print(f"[{s()}] [experiment]   {label:<15} = {val}")
    else:
        bv = bm.get(label)
        if isinstance(pv, (int, float)) and isinstance(bv, (int, float)):
            d = pv - bv
            delta = f"{d:+.3f}"
        else:
            delta = "N/A"
        print(f"[{s()}] [experiment]   {label:<15} = {val} (delta = {delta})")
PY
}

# =====================================================================
# Helper: final comparison table (Phase 5)
# =====================================================================
print_final_table() {
    local baseline_json="$1"
    local quantized_json="$2"

    BASELINE="$baseline_json" QUANTIZED="$quantized_json" \
        MODEL_NAME_ENV="$MODEL_NAME" QUANT_K_ENV="$QUANT_K" python - <<'PY'
import json, os, time

with open(os.environ["BASELINE"]) as f:
    bm = json.load(f)["key_metrics"]
with open(os.environ["QUANTIZED"]) as f:
    qm = json.load(f)["key_metrics"]
model = os.environ["MODEL_NAME_ENV"]
qk = os.environ["QUANT_K_ENV"]

labels = ["wikitext2_ppl", "c4_ppl", "arc_challenge", "arc_easy",
          "boolq", "piqa", "hellaswag", "winogrande",
          "gsm8k_5shot", "ifeval_strict"]

def s():
    return time.strftime("%H:%M:%S")

bar = "=" * 60
short_model = model.split("/")[-1]
print(f"[{s()}] [experiment] {bar}")
print(f"[{s()}] [experiment] FINAL COMPARISON: baseline vs quantized ({short_model}, K={qk})")
print(f"[{s()}] [experiment] {bar}")
print(f"[{s()}] [experiment]   {'metric':<16} {'baseline':<10} {'quantized':<10} {'delta':<10}")
for label in labels:
    bv = bm.get(label); qv = qm.get(label)
    bv_s = f"{bv:.3f}" if isinstance(bv, (int, float)) else "N/A"
    qv_s = f"{qv:.3f}" if isinstance(qv, (int, float)) else "N/A"
    if isinstance(bv, (int, float)) and isinstance(qv, (int, float)):
        d_s = f"{(qv - bv):+.3f}"
    else:
        d_s = "N/A"
    print(f"[{s()}] [experiment]   {label:<16} {bv_s:<10} {qv_s:<10} {d_s:<10}")
print(f"[{s()}] [experiment] {bar}")
PY
}

# =====================================================================
# Header
# =====================================================================
log "============================================================"
log "Llama-2-7B true-accum Fisher experiment"
log "WORK_DIR=$WORK_DIR"
log "REPO_ROOT=$REPO_ROOT"
log "MODEL=$MODEL_NAME  K=$QUANT_K  num_layers=$NUM_LAYERS"
log "tasks=$TASKS_LIST"
log "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
[[ -n "$SKIP_BASELINE" ]] && log "CLI: --skip-baseline"
[[ -n "$SKIP_FISHER" ]]   && log "CLI: --skip-fisher"
[[ -n "$SKIP_QUANTIZE" ]] && log "CLI: --skip-quantize"
[[ -n "$FORCE_BASELINE" ]] && log "CLI: --force-baseline"
log "============================================================"

OVERALL_T0=$(date +%s)

# =====================================================================
# Phase 1 — BASELINE EVAL
# =====================================================================
PHASE="1_baseline"
log ""
log "------------------------- Phase 1: BASELINE EVAL -------------------------"

if [[ -n "$SKIP_BASELINE" ]]; then
    if [[ ! -f "$BASELINE_JSON" ]]; then
        log "FATAL: --skip-baseline requested but $BASELINE_JSON does not exist"
        exit 1
    fi
    log "phase 1: SKIP (--skip-baseline) — using cached $BASELINE_JSON"
    PHASE_TIME[1_baseline]=0
elif [[ -f "$BASELINE_JSON" && -z "$FORCE_BASELINE" ]]; then
    log "phase 1: cache HIT — using existing $BASELINE_JSON (pass --force-baseline to recompute)"
    PHASE_TIME[1_baseline]=0
else
    [[ -n "$FORCE_BASELINE" ]] && log "phase 1: cache invalidated by --force-baseline"
    p1_t0=$(date +%s)
    PPL_TMP="$WORK_DIR/_tmp_ppl_baseline.json"
    LM_TMP="$WORK_DIR/_tmp_lm_baseline.json"
    rm -f "$PPL_TMP" "$LM_TMP"

    log "phase 1: running eval_ppl on $MODEL_NAME"
    pushd "$REPO_ROOT" >/dev/null
    python3 "$EVAL_PPL_PY" \
        --hf_path "$MODEL_NAME" \
        --output_json "$PPL_TMP" \
        2>&1 | tee -a "$LOGS_DIR/phase1_baseline_ppl.log"
    popd >/dev/null

    log "phase 1: running eval_zeroshot on $MODEL_NAME (tasks: $TASKS_LIST)"
    pushd "$REPO_ROOT" >/dev/null
    python3 "$EVAL_LM_PY" \
        --hf_path "$MODEL_NAME" \
        --tokenizer "$MODEL_NAME" \
        --tasks "$TASKS_LIST" \
        --num_fewshot -1 \
        --batch_size "$EVAL_BATCH_SIZE" \
        --output_json "$LM_TMP" \
        2>&1 | tee -a "$LOGS_DIR/phase1_baseline_lm_eval.log"
    popd >/dev/null

    log "phase 1: merging into $BASELINE_JSON"
    merge_eval_json "baseline" "$MODEL_NAME" "$PPL_TMP" "$LM_TMP" "$BASELINE_JSON" \
        2>&1 | tee -a "$LOGS_DIR/phase1_merge.log"
    rm -f "$PPL_TMP" "$LM_TMP"

    PHASE_TIME[1_baseline]=$(( $(date +%s) - p1_t0 ))
    log "phase 1: DONE in ${PHASE_TIME[1_baseline]}s"
fi

print_metrics_block "BASELINE" "$BASELINE_JSON" ""

# =====================================================================
# Phase 2 — FISHER FACTORS
# =====================================================================
PHASE="2_fisher"
log ""
log "------------------------- Phase 2: FISHER FACTORS -------------------------"

if [[ -n "$SKIP_FISHER" ]]; then
    log "phase 2: SKIP (--skip-fisher)"
    PHASE_TIME[2_fisher]=0
else
    p2_t0=$(date +%s)
    log "phase 2: invoking $FISHER_WRAPPER"

    MODEL_NAME="$MODEL_NAME" \
    NUM_LAYERS="$NUM_LAYERS" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    DATASET_SIZE="$DATASET_SIZE" \
    GRAD_ACCUM="$GRAD_ACCUM" \
    MAX_LENGTH="$MAX_LENGTH" \
    PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
    ATTN_IMPL="$ATTN_IMPL" \
    COLLECT_SCRIPT="$COLLECT_SCRIPT" \
    WORK_DIR="$WORK_DIR" \
        bash "$FISHER_WRAPPER" 2>&1 | tee -a "$LOGS_DIR/phase2_fisher_wrapper.log"

    PHASE_TIME[2_fisher]=$(( $(date +%s) - p2_t0 ))
    log "phase 2: DONE in ${PHASE_TIME[2_fisher]}s"
fi

# Validate factors count regardless of skip
n_factors=$(find "$FACTORS_DIR" -maxdepth 1 -name '*.safetensors' -type f | wc -l)
expected=$(( NUM_LAYERS * 7 ))
if (( n_factors != expected )); then
    log "FATAL: factors/ has $n_factors files, expected $expected (NUM_LAYERS=$NUM_LAYERS × 7 projs)"
    exit 1
fi
log "phase 2: $n_factors factor files present (matches expected $expected)"

# =====================================================================
# Phase 3a — QUANTIZE
# =====================================================================
PHASE="3a_quantize"
log ""
log "------------------------- Phase 3a: QUANTIZE (K=$QUANT_K) -------------------------"

PHASE3A_FLAG="$WORK_DIR/done_phase_3a_quantize.flag"
if [[ -n "$SKIP_QUANTIZE" ]]; then
    log "phase 3a: SKIP (--skip-quantize)"
    PHASE_TIME[3a_quantize]=0
elif [[ -f "$PHASE3A_FLAG" ]]; then
    log "phase 3a: SKIP (flag present)"
    PHASE_TIME[3a_quantize]=0
else
    p3a_t0=$(date +%s)
    pushd "$REPO_ROOT" >/dev/null
    python3 "$QUANT_PY" \
        --base_model "$MODEL_NAME" \
        --hess_path "$FACTORS_DIR" \
        --save_path "$QUANTIZED_DIR" \
        --codebook bitshift \
        --scale_override 0.9 \
        --ft_epochs 0 \
        --decode_mode quantlut_sym \
        --tlut_bits "$QUANT_TLUT_BITS" \
        --L "$QUANT_L" \
        --K "$QUANT_K" \
        --V "$QUANT_V" \
        --td_x "$QUANT_TD_X" \
        --td_y "$QUANT_TD_Y" \
        2>&1 | tee -a "$LOGS_DIR/phase3_quantize.log"
    popd >/dev/null
    touch "$PHASE3A_FLAG"
    PHASE_TIME[3a_quantize]=$(( $(date +%s) - p3a_t0 ))
    log "phase 3a: DONE in ${PHASE_TIME[3a_quantize]}s"
fi

# =====================================================================
# Phase 3b — HFize
# =====================================================================
PHASE="3b_hfize"
log ""
log "------------------------- Phase 3b: HFize -------------------------"

PHASE3B_FLAG="$WORK_DIR/done_phase_3b_hfize.flag"
if [[ -n "$SKIP_QUANTIZE" ]]; then
    log "phase 3b: SKIP (--skip-quantize)"
    PHASE_TIME[3b_hfize]=0
elif [[ -f "$PHASE3B_FLAG" ]]; then
    log "phase 3b: SKIP (flag present)"
    PHASE_TIME[3b_hfize]=0
else
    p3b_t0=$(date +%s)
    pushd "$REPO_ROOT" >/dev/null
    python3 "$HFIZE_PY" \
        --quantized_path "$QUANTIZED_DIR" \
        --hf_output_path "$HF_DIR" \
        2>&1 | tee -a "$LOGS_DIR/phase3_hfize.log"
    popd >/dev/null
    touch "$PHASE3B_FLAG"
    PHASE_TIME[3b_hfize]=$(( $(date +%s) - p3b_t0 ))
    log "phase 3b: DONE in ${PHASE_TIME[3b_hfize]}s"
fi

if [[ ! -f "$HF_DIR/config.json" ]]; then
    log "FATAL: $HF_DIR/config.json missing — quantize/hfize did not produce a usable model"
    exit 1
fi
log "phase 3: HF model at $HF_DIR (config.json present)"

# =====================================================================
# Phase 4 — QUANTIZED EVAL
# =====================================================================
PHASE="4_quantized_eval"
log ""
log "------------------------- Phase 4: QUANTIZED EVAL -------------------------"
log "phase 4: always recomputed (no cache)"

p4_t0=$(date +%s)
PPL_TMP="$WORK_DIR/_tmp_ppl_quantized.json"
LM_TMP="$WORK_DIR/_tmp_lm_quantized.json"
rm -f "$PPL_TMP" "$LM_TMP"

log "phase 4: running eval_ppl on $HF_DIR (--manifest)"
pushd "$REPO_ROOT" >/dev/null
python3 "$EVAL_PPL_PY" \
    --hf_path "$HF_DIR" \
    --manifest \
    --output_json "$PPL_TMP" \
    2>&1 | tee -a "$LOGS_DIR/phase4_quantized_ppl.log"
popd >/dev/null

log "phase 4: running eval_zeroshot on $HF_DIR (--manifest_model)"
pushd "$REPO_ROOT" >/dev/null
python3 "$EVAL_LM_PY" \
    --hf_path "$HF_DIR" \
    --tokenizer "$MODEL_NAME" \
    --tasks "$TASKS_LIST" \
    --num_fewshot -1 \
    --batch_size "$EVAL_BATCH_SIZE" \
    --manifest_model \
    --output_json "$LM_TMP" \
    2>&1 | tee -a "$LOGS_DIR/phase4_quantized_lm_eval.log"
popd >/dev/null

log "phase 4: merging into $QUANTIZED_JSON"
merge_eval_json "quantized" "$HF_DIR" "$PPL_TMP" "$LM_TMP" "$QUANTIZED_JSON" \
    2>&1 | tee -a "$LOGS_DIR/phase4_merge.log"
rm -f "$PPL_TMP" "$LM_TMP"

PHASE_TIME[4_quantized_eval]=$(( $(date +%s) - p4_t0 ))
log "phase 4: DONE in ${PHASE_TIME[4_quantized_eval]}s"

print_metrics_block "QUANTIZED" "$QUANTIZED_JSON" "$BASELINE_JSON"

# =====================================================================
# Phase 5 — COMPARISON SUMMARY
# =====================================================================
PHASE="5_summary"
log ""
log "------------------------- Phase 5: COMPARISON SUMMARY -------------------------"

print_final_table "$BASELINE_JSON" "$QUANTIZED_JSON"

OVERALL_DT=$(( $(date +%s) - OVERALL_T0 ))

log "phase timings:"
for ph in 1_baseline 2_fisher 3a_quantize 3b_hfize 4_quantized_eval; do
    if [[ -n "${PHASE_TIME[$ph]:-}" ]]; then
        printf '[%s] [experiment]   phase %-20s %ds\n' "$(stamp)" "$ph:" "${PHASE_TIME[$ph]}"
    fi
done
log "  total experiment time: ${OVERALL_DT}s"
log "============================================================"
log "EXPERIMENT COMPLETE"
log "  baseline: $BASELINE_JSON"
log "  quantized: $QUANTIZED_JSON"
log "  factors:  $FACTORS_DIR"
log "  hf model: $HF_DIR"
log "============================================================"
