#!/usr/bin/env bash
# ============================================================
# run_demo.sh — Quick launcher for DeceptiNet camera demo
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# --- defaults ---
CHECKPOINT="${SCRIPT_DIR}/checkpoints/best_model.pt"
DINOV3_WEIGHTS="${SCRIPT_DIR}/checkpoints/dinov3_vits16_pretrain.pth"
CAMERA_ID=0
DEVICE=""
INTERVAL=0.5
MODE="live"   # live | video

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --checkpoint PATH   Path to DeceptiNet checkpoint   (default: checkpoints/best_model.pt)
  --dinov3 PATH       Path to DINOv3 weights          (default: checkpoints/dinov3_vits16_pretrain.pth)
  --camera ID         Webcam device ID                (default: 0)
  --device DEVICE     cuda or cpu                     (default: auto-detect)
  --interval SECS     Inference frequency in seconds  (default: 0.5)
  --video FILE        Process a video file instead of live camera
  -h, --help          Show this help

Examples:
  $0                                         # live webcam, auto-detect GPU
  $0 --device cpu                            # force CPU
  $0 --video recordings/demo.mp4             # process a saved video
  $0 --camera 1 --interval 1.0              # second webcam, 1s interval
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dinov3)     DINOV3_WEIGHTS="$2"; shift 2 ;;
        --camera)     CAMERA_ID="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --interval)   INTERVAL="$2"; shift 2 ;;
        --video)      MODE="video"; VIDEO_FILE="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Locate Python
if command -v python3 &>/dev/null; then PYTHON=python3
elif command -v python &>/dev/null; then PYTHON=python
else echo "❌  Python not found. Install Python 3.8+"; exit 1; fi

# Validate checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌  Checkpoint not found: $CHECKPOINT"
    echo "   Place your .pt file in checkpoints/ or pass --checkpoint PATH"
    exit 1
fi

cd "$SCRIPT_DIR"

if [ "$MODE" = "video" ]; then
    if [ -z "${VIDEO_FILE:-}" ]; then echo "❌  --video requires a file path"; exit 1; fi
    echo "▶  Processing video: $VIDEO_FILE"
    ARGS=("process_video.py" "--video" "$VIDEO_FILE" "--checkpoint" "$CHECKPOINT" "--interval" "$INTERVAL")
else
    echo "▶  Starting live camera demo (camera $CAMERA_ID)"
    ARGS=("camera_demo.py" "--checkpoint" "$CHECKPOINT" "--camera" "$CAMERA_ID" "--interval" "$INTERVAL")
fi

[ -n "$DEVICE" ]            && ARGS+=("--device" "$DEVICE")
[ -f "$DINOV3_WEIGHTS" ]    && export DINOV3_WEIGHTS

echo "   Checkpoint : $CHECKPOINT"
echo "   DINOv3     : ${DINOV3_WEIGHTS}"
echo "   Device     : ${DEVICE:-auto}"
echo ""
exec "$PYTHON" "${ARGS[@]}"
