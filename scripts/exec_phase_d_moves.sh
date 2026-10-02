#!/usr/bin/env bash
# Phase D Execution: ACTUAL FILE MOVES
# WARNING: This script performs real I/O operations.
# Run only after reviewing Phase ABC plan and confirming validation checklist.

set -e
cd /home/mmahaut/projects/paramem || exit 1

DRY_RUN="${1:-yes}"  # Default: dry-run. Pass "execute" to actually move files.

if [ "$DRY_RUN" = 'yes' ]; then
  echo "=== PHASE D DRY RUN ==="
  echo "To execute moves, run: $0 execute"
  echo ""
  MKDIRFLAG='-vn'
  MVFLAG='-vn'
else
  echo "=== PHASE D EXECUTION (ACTUAL MOVES) ==="
  MKDIRFLAG='-v'
  MVFLAG='-v'
fi

echo "Step 1: Create missing destination directories..."
while IFS=',' read -r run_id family mode dataset status action src_path dst_path reason_note; do
  if [ "$run_id" = 'run_id' ]; then continue; fi
  dir=$(dirname "$dst_path")
  mkdir $MKDIRFLAG -p "$dir" 2>/dev/null || true
done < results/index/move_plan.csv

echo "Step 2: Move logs per Phase B plan..."
count=0
while IFS=',' read -r run_id family mode dataset status action src_path dst_path reason_note; do
  if [ "$run_id" = 'run_id' ]; then continue; fi
  if [ -f "$src_path" ]; then
    if [ "$DRY_RUN" = 'execute' ]; then
      mv $MVFLAG "$src_path" "$dst_path" 2>/dev/null || true
    else
      echo "[DRY] mv $src_path $dst_path"
    fi
    count=$((count + 1))
  fi
done < results/index/move_plan.csv
echo "Phase B: moved/would-move $count log files"

echo "Step 3: Move artifacts per Phase C plan..."
count=0
while IFS=',' read -r artifact type current_path destination action size_mb reason; do
  if [ "$artifact" = 'artifact' ]; then continue; fi
  
  # Skip root-level logs (already handled in Phase B)
  if [ "$type" = 'log' ]; then continue; fi
  
  if [ -f "$current_path" ]; then
    dir=$(dirname "$destination")
    mkdir $MKDIRFLAG -p "$dir" 2>/dev/null || true
    
    if [ "$DRY_RUN" = 'execute' ]; then
      mv $MVFLAG "$current_path" "$destination" 2>/dev/null || true
    else
      echo "[DRY] mv $current_path $destination"
    fi
    count=$((count + 1))
  fi
done < results/index/phase_c_artifact_plan.csv
echo "Phase C: moved/would-move $count artifact files"

if [ "$DRY_RUN" = 'yes' ]; then
  echo ""
  echo "This was a DRY RUN. Review output above, then run:"
  echo "  $0 execute"
  echo ""
  echo "Or return to user for approval first."
else
  echo ""
  echo "=== MOVES COMPLETE ==="
  echo "Verify results:"
  echo "  ls -lh results/final/"
  echo "  ls -lh results/archive/"
  echo "  ls -lh results/failed_runs/"
fi
