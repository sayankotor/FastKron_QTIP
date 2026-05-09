#!/usr/bin/env bash
#
# run_chunked_fisher.sh — orchestrate chunked collection of Fisher gradients
# and Kronecker factors for large LLMs (e.g. Qwen2.5-32B).
#
# Pipeline per chunk:
#   1. clean grads/chunk_K
#   2. collect_fisher_weights.py → grads/chunk_K/grad_step_*.safetensors
#   3. verify expected step count (and non-empty)
#   4. get_kron_factors_llama.py → factors/model_layers_*_proj.safetensors
#   5. verify factor file integrity (XF/YF/s present, finite, square)
#   6. delete grads/chunk_K (preserve only factors)
#   7. touch done_chunk_K.flag
#
# Re-run: skip chunks with done_chunk_K.flag. Force re-run: rm flag.
# CLI: --start-chunk N | --only-chunk N | --dry-run | --help

set -euo pipefail
umask 022

# ---------- helpers ----------
stamp() { date +%H:%M:%S; }
log() { echo "[$(stamp)] [wrapper] $*"; }

# ---------- env vars ----------
: "${WORK_DIR:?WORK_DIR must be set (e.g. /workspace.../qwen25_32b_run1)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B}"
NUM_LAYERS="${NUM_LAYERS:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
DATASET_PATH="${DATASET_PATH:-/workspace-SR004.nfs2/data/fineweb/sample/66K}"
DATASET_SIZE="${DATASET_SIZE:-38400}"
GRAD_ACCUM="${GRAD_ACCUM:-384}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-3096}"
LR="${LR:-1e-4}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
TOP_K="${TOP_K:-2}"
GRAD_CHUNK_SIZE="${GRAD_CHUNK_SIZE:-4}"
NUM_DEVICES="${NUM_DEVICES:-1}"
EXPECTED_STEPS="${EXPECTED_STEPS:-$(( DATASET_SIZE / (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM) ))}"

# ---------- repo paths ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_PY="${REPO_ROOT}/kronfwsvd/collect_fisher_weights.py"
KRON_PY="${REPO_ROOT}/kronfwsvd/get_kron_factors_llama.py"

# ---------- ERR trap (CHUNK_IDX set in main loop) ----------
CHUNK_IDX="pre-loop"
trap 'log "FAILED at chunk ${CHUNK_IDX}, line ${LINENO}, exit $?"; exit 1' ERR

# ---------- CLI parser ----------
START_CHUNK="${START_CHUNK:-}"
ONLY_CHUNK="${ONLY_CHUNK:-}"
DRY_RUN="${DRY_RUN:-}"

usage() {
    cat <<EOF
usage: $0 [--start-chunk N] [--only-chunk N] [--dry-run] [--help]

Mandatory env: WORK_DIR

Optional env (with defaults):
  MODEL_NAME=${MODEL_NAME}
  NUM_LAYERS=${NUM_LAYERS}        CHUNK_SIZE=${CHUNK_SIZE}
  EXPECTED_STEPS=${EXPECTED_STEPS}     (auto = DATASET_SIZE / (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM))
  DATASET_PATH=${DATASET_PATH}
  DATASET_SIZE=${DATASET_SIZE}    GRAD_ACCUM=${GRAD_ACCUM}
  PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}     MAX_LENGTH=${MAX_LENGTH}    LR=${LR}
  ATTN_IMPL=${ATTN_IMPL}
  TOP_K=${TOP_K}     GRAD_CHUNK_SIZE=${GRAD_CHUNK_SIZE}    NUM_DEVICES=${NUM_DEVICES}

CLI:
  --start-chunk N    start from chunk N (lower chunks unconditionally skipped)
  --only-chunk N     process only chunk N (still respects done_chunk_N.flag)
  --dry-run          print commands without executing
  --help             show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-chunk) START_CHUNK="$2"; shift 2 ;;
        --only-chunk)  ONLY_CHUNK="$2";  shift 2 ;;
        --dry-run)     DRY_RUN=1;        shift ;;
        --help|-h)     usage; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$START_CHUNK" && -n "$ONLY_CHUNK" ]]; then
    echo "--start-chunk and --only-chunk are mutually exclusive" >&2
    exit 2
fi

# ---------- validate env ----------
validate_env() {
    local var
    for var in NUM_LAYERS CHUNK_SIZE EXPECTED_STEPS GRAD_ACCUM MAX_LENGTH \
               TOP_K GRAD_CHUNK_SIZE NUM_DEVICES PER_DEVICE_BATCH_SIZE DATASET_SIZE; do
        if [[ ! "${!var}" =~ ^[0-9]+$ ]]; then
            echo "env $var must be a non-negative integer, got: ${!var}" >&2
            exit 2
        fi
    done
    (( CHUNK_SIZE > 0 )) || { echo "CHUNK_SIZE must be > 0" >&2; exit 2; }
    (( CHUNK_SIZE <= NUM_LAYERS )) || { echo "CHUNK_SIZE ($CHUNK_SIZE) > NUM_LAYERS ($NUM_LAYERS)" >&2; exit 2; }
    (( EXPECTED_STEPS > 0 )) || { echo "EXPECTED_STEPS computed as 0; check DATASET_SIZE / (BATCH * GRAD_ACCUM)" >&2; exit 2; }
    [[ -d "$DATASET_PATH" ]] || { echo "DATASET_PATH not a directory: $DATASET_PATH" >&2; exit 2; }
    [[ -f "$COLLECT_PY" ]] || { echo "missing $COLLECT_PY" >&2; exit 2; }
    [[ -f "$KRON_PY" ]] || { echo "missing $KRON_PY" >&2; exit 2; }
    if [[ -n "$START_CHUNK" ]] && [[ ! "$START_CHUNK" =~ ^[0-9]+$ ]]; then
        echo "--start-chunk must be a non-negative integer, got: $START_CHUNK" >&2; exit 2
    fi
    if [[ -n "$ONLY_CHUNK" ]] && [[ ! "$ONLY_CHUNK" =~ ^[0-9]+$ ]]; then
        echo "--only-chunk must be a non-negative integer, got: $ONLY_CHUNK" >&2; exit 2
    fi
    mkdir -p "$WORK_DIR"/{grads,factors,logs}
}
validate_env

# ---------- chunk arithmetic ----------
TOTAL_CHUNKS=$(( (NUM_LAYERS + CHUNK_SIZE - 1) / CHUNK_SIZE ))

# ---------- timing / status (indexed by chunk number) ----------
declare -a CHUNK_COLLECT_TIME
declare -a CHUNK_KRON_TIME
declare -a CHUNK_RANGE_STR
declare -a CHUNK_STATUS  # "done" | "skipped" | "cli-skip"

# ---------- dry-run wrapper ----------
run_or_dry() {
    if [[ -n "$DRY_RUN" ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# ---------- phase: verify grads ----------
verify_grads() {
    local idx="$1"
    local chunk_dir="${WORK_DIR}/grads/chunk_${idx}"
    if [[ -n "$DRY_RUN" ]]; then
        log "chunk ${idx}: [DRY-RUN] would verify ${EXPECTED_STEPS} grad files in ${chunk_dir}"
        return 0
    fi
    local actual=0 empty=0 f
    for f in "$chunk_dir"/grad_step_*.safetensors; do
        [[ -e "$f" ]] || continue
        if [[ -s "$f" ]]; then
            actual=$((actual + 1))
        else
            empty=$((empty + 1))
            log "chunk ${idx}: empty grad file: $f"
        fi
    done
    if (( actual != EXPECTED_STEPS )) || (( empty > 0 )); then
        log "chunk ${idx}: expected ${EXPECTED_STEPS} non-empty grad files, got ${actual} (empty: ${empty})"
        return 1
    fi
    log "chunk ${idx}: ${actual} grad files verified"
}

# ---------- phase: verify factors ----------
verify_factors() {
    local idx="$1" start="$2" end="$3"
    if [[ -n "$DRY_RUN" ]]; then
        log "chunk ${idx}: [DRY-RUN] would verify $(( (end - start) * 7 )) factor files"
        return 0
    fi
    local pairs=( "self_attn:q" "self_attn:k" "self_attn:v" "self_attn:o" \
                  "mlp:gate" "mlp:up" "mlp:down" )
    local missing=0 i pair block proj f
    for (( i=start; i<end; i++ )); do
        for pair in "${pairs[@]}"; do
            block="${pair%:*}"
            proj="${pair#*:}"
            f="${WORK_DIR}/factors/model_layers_${i}_${block}_${proj}_proj.safetensors"
            if [[ ! -s "$f" ]]; then
                log "chunk ${idx}: missing or empty factor file: $f"
                missing=$((missing + 1))
            fi
        done
    done
    if (( missing > 0 )); then
        return 1
    fi

    LAYER_START="$start" LAYER_END="$end" CHUNK_IDX_FOR_PY="$idx" \
        FACTORS_DIR="${WORK_DIR}/factors" python - <<'PY'
import os, sys, torch
from pathlib import Path
from safetensors import safe_open

start = int(os.environ["LAYER_START"])
end = int(os.environ["LAYER_END"])
chunk = os.environ["CHUNK_IDX_FOR_PY"]
factors = Path(os.environ["FACTORS_DIR"])

projs = [("self_attn", "q"), ("self_attn", "k"), ("self_attn", "v"),
         ("self_attn", "o"), ("mlp", "gate"), ("mlp", "up"), ("mlp", "down")]
required_keys = {"XF", "YF", "s"}

bad = []
for i in range(start, end):
    for block, proj in projs:
        f = factors / f"model_layers_{i}_{block}_{proj}_proj.safetensors"
        try:
            with safe_open(str(f), framework="pt", device="cpu") as h:
                keys = set(h.keys())
                if not required_keys.issubset(keys):
                    bad.append(f"{f.name}: missing keys, have {sorted(keys)}, need {sorted(required_keys)}")
                    continue
                ok_xfyf = True
                for k in ("XF", "YF"):
                    t = h.get_tensor(k)
                    if t.dim() != 2 or t.shape[0] != t.shape[1]:
                        bad.append(f"{f.name}:{k} bad shape {tuple(t.shape)}")
                        ok_xfyf = False
                        break
                    if not torch.isfinite(t).all().item():
                        bad.append(f"{f.name}:{k} contains NaN/Inf")
                        ok_xfyf = False
                        break
                if not ok_xfyf:
                    continue
                s_t = h.get_tensor("s")
                if s_t.numel() < 1:
                    bad.append(f"{f.name}:s is empty (numel=0)")
                elif not torch.isfinite(s_t).all().item():
                    bad.append(f"{f.name}:s contains NaN/Inf")
        except Exception as e:
            bad.append(f"{f.name}: read error: {e}")

if bad:
    for line in bad:
        print(f"[verify] {line}", flush=True)
    sys.exit(1)

n = (end - start) * len(projs)
print(f"[verify] chunk {chunk} integrity OK ({end - start} layers x {len(projs)} projs = {n} files)", flush=True)
PY
    log "chunk ${idx}: factor files verified"
}

# ---------- phase: process one chunk ----------
process_chunk() {
    local idx="$1" start="$2" end="$3"
    local hdr="chunk ${idx}/${TOTAL_CHUNKS} ${CHUNK_RANGE_STR[$idx]}"
    local flag="${WORK_DIR}/done_chunk_${idx}.flag"

    if [[ -f "$flag" ]]; then
        log "${hdr}: SKIP (flag present)"
        CHUNK_STATUS[$idx]="skipped"
        return 0
    fi

    local chunk_grad="${WORK_DIR}/grads/chunk_${idx}"
    local collect_log="${WORK_DIR}/logs/collect_chunk_${idx}.log"
    local kron_log="${WORK_DIR}/logs/kron_chunk_${idx}.log"

    log "${hdr}: starting collect"
    run_or_dry rm -rf "$chunk_grad"
    run_or_dry mkdir -p "$chunk_grad"

    local collect_t0 collect_dt
    collect_t0=$(date +%s)
    if [[ -n "$DRY_RUN" ]]; then
        echo "[DRY-RUN] python ${COLLECT_PY} --model_name ${MODEL_NAME} --grad_dir ${chunk_grad} --layer_start ${start} --layer_end ${end} ..."
    else
        python "$COLLECT_PY" \
            --model_name "$MODEL_NAME" \
            --grad_dir "$chunk_grad" \
            --layer_start "$start" \
            --layer_end "$end" \
            --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --lr "$LR" \
            --dataset_path "$DATASET_PATH" \
            --dataset_size "$DATASET_SIZE" \
            --attn_impl "$ATTN_IMPL" \
            2>&1 | tee -a "$collect_log"
    fi
    collect_dt=$(( $(date +%s) - collect_t0 ))
    log "${hdr}: collect done in ${collect_dt}s"

    log "${hdr}: verifying grads (expected ${EXPECTED_STEPS})"
    verify_grads "$idx"

    log "${hdr}: starting kron"
    local kron_t0 kron_dt
    kron_t0=$(date +%s)
    if [[ -n "$DRY_RUN" ]]; then
        echo "[DRY-RUN] python ${KRON_PY} --model_name ${MODEL_NAME} --grad_dir ${chunk_grad} --kron_dir ${WORK_DIR}/factors --layer_start ${start} --layer_end ${end} ..."
    else
        python "$KRON_PY" \
            --model_name "$MODEL_NAME" \
            --grad_dir "$chunk_grad" \
            --kron_dir "${WORK_DIR}/factors" \
            --layer_start "$start" \
            --layer_end "$end" \
            --top_k "$TOP_K" \
            --grad_chunk_size "$GRAD_CHUNK_SIZE" \
            --num_devices "$NUM_DEVICES" \
            2>&1 | tee -a "$kron_log"
    fi
    kron_dt=$(( $(date +%s) - kron_t0 ))
    log "${hdr}: kron done in ${kron_dt}s"

    log "${hdr}: verifying $(( (end - start) * 7 )) factor files"
    verify_factors "$idx" "$start" "$end"

    log "${hdr}: cleaning grads"
    run_or_dry rm -rf "$chunk_grad"

    log "${hdr}: DONE — flag written"
    run_or_dry touch "$flag"

    CHUNK_COLLECT_TIME[$idx]=$collect_dt
    CHUNK_KRON_TIME[$idx]=$kron_dt
    CHUNK_STATUS[$idx]="done"
}

# ---------- main ----------
log "starting | WORK_DIR=${WORK_DIR} model=${MODEL_NAME} num_layers=${NUM_LAYERS} chunk_size=${CHUNK_SIZE} total_chunks=${TOTAL_CHUNKS}"
log "config  | dataset=${DATASET_PATH} (size=${DATASET_SIZE}) batch=${PER_DEVICE_BATCH_SIZE} grad_accum=${GRAD_ACCUM} expected_steps=${EXPECTED_STEPS}"
log "config  | top_k=${TOP_K} grad_chunk_size=${GRAD_CHUNK_SIZE} num_devices=${NUM_DEVICES} attn_impl=${ATTN_IMPL}"
[[ -n "$DRY_RUN" ]] && log "DRY-RUN mode: no commands will be executed"
[[ -n "$START_CHUNK" ]] && log "CLI: --start-chunk ${START_CHUNK}"
[[ -n "$ONLY_CHUNK" ]] && log "CLI: --only-chunk ${ONLY_CHUNK}"

OVERALL_T0=$(date +%s)

for (( CHUNK_IDX=0; CHUNK_IDX<TOTAL_CHUNKS; CHUNK_IDX++ )); do
    LAYER_START=$(( CHUNK_IDX * CHUNK_SIZE ))
    LAYER_END=$(( LAYER_START + CHUNK_SIZE ))
    if (( LAYER_END > NUM_LAYERS )); then
        LAYER_END=$NUM_LAYERS
    fi
    CHUNK_RANGE_STR[$CHUNK_IDX]="[layers ${LAYER_START}..${LAYER_END})"

    if [[ -n "$START_CHUNK" ]] && (( CHUNK_IDX < START_CHUNK )); then
        log "chunk ${CHUNK_IDX}/${TOTAL_CHUNKS} ${CHUNK_RANGE_STR[$CHUNK_IDX]}: SKIP (--start-chunk ${START_CHUNK})"
        CHUNK_STATUS[$CHUNK_IDX]="cli-skip"
        continue
    fi
    if [[ -n "$ONLY_CHUNK" ]] && (( CHUNK_IDX != ONLY_CHUNK )); then
        CHUNK_STATUS[$CHUNK_IDX]="cli-skip"
        continue
    fi

    process_chunk "$CHUNK_IDX" "$LAYER_START" "$LAYER_END"
done

OVERALL_DT=$(( $(date +%s) - OVERALL_T0 ))

# ---------- final summary ----------
echo
log "all chunks done | total time=${OVERALL_DT}s"
for (( i=0; i<TOTAL_CHUNKS; i++ )); do
    if [[ -z "${CHUNK_STATUS[$i]:-}" ]]; then
        continue
    fi
    case "${CHUNK_STATUS[$i]}" in
        done)
            log "  chunk $i ${CHUNK_RANGE_STR[$i]}: collect=${CHUNK_COLLECT_TIME[$i]:-?}s kron=${CHUNK_KRON_TIME[$i]:-?}s"
            ;;
        skipped)
            log "  chunk $i ${CHUNK_RANGE_STR[$i]}: SKIPPED (flag present)"
            ;;
        cli-skip)
            ;;
    esac
done

if [[ -z "$DRY_RUN" ]]; then
    n_factors=$(find "$WORK_DIR/factors" -maxdepth 1 -name '*.safetensors' -type f | wc -l)
    expected_total=$(( NUM_LAYERS * 7 ))
    log "factor files in ${WORK_DIR}/factors: ${n_factors} (full-model expected ${expected_total})"
fi
