#!/usr/bin/env bash
cd /home/mmahaut/projects/paramem || exit 1

manifest='results/index/run_manifest.csv'
moveplan='results/index/move_plan.csv'

# Read manifest and identify canonical runs (most recent success per family+dataset+mode)
declare -A canonical

echo 'run_id,family,mode,dataset,status,action,src_path,dst_path,reason' > "$moveplan"

# Parse manifest to find canonical runs
while IFS=',' read -r run_id family mode dataset status reason has_wandb output_path log_err log_out kept_as; do
  if [ "$run_id" = 'run_id' ]; then continue; fi
  
  key="$family|$mode|$dataset"
  
  # Mark as canonical if it's a success and we haven't seen this key yet (most recent = first in processing order)
  if [ "$status" = 'success' ] && [ -z "${canonical[$key]}" ]; then
    canonical["$key"]="$run_id"
  fi
done < "$manifest"

echo "=== Identified ${#canonical[@]} canonical runs ==="
for key in "${!canonical[@]}"; do
  echo "  $key => run_id=${canonical[$key]}"
done

# Now generate move plan for all files
while IFS=',' read -r run_id family mode dataset status reason has_wandb output_path log_err log_out kept_as; do
  if [ "$run_id" = 'run_id' ]; then continue; fi
  
  key="$family|$mode|$dataset"
  
  # Determine action based on status and canonical flag
  action='archive'
  reason_note='superseded'
  
  if [ "$status" = 'success' ] && [ "${canonical[$key]}" = "$run_id" ]; then
    action='keep_final'
    reason_note='canonical_success'
  elif [ "$status" = 'failed' ]; then
    action='failed_logs'
    reason_note="$reason"
  elif [ "$status" = 'unknown' ]; then
    action='archive'
    reason_note='unknown_status'
  fi
  
  # Determine destination path
  if [ -n "$log_err" ] && [ -f "$log_err" ]; then
    base=$(basename "$log_err")
    
    case "$action" in
      keep_final)
        dst="results/final/logs/${family}/${base}"
        ;;
      failed_logs)
        if [[ "$log_err" =~ ^slurm_logs ]]; then
          dst="results/failed_runs/slurm_logs/${base}"
        else
          dst="results/failed_runs/root_logs/${base}"
        fi
        ;;
      archive)
        if [[ "$log_err" =~ ^slurm_logs ]]; then
          dst="results/archive/old_logs/slurm_logs/${base}"
        else
          dst="results/archive/old_logs/root/${base}"
        fi
        ;;
    esac
    
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$run_id" "$family" "$mode" "$dataset" "$status" "$action" "$log_err" "$dst" "$reason_note" >> "$moveplan"
  fi
done < "$manifest"

echo "Move plan generated: $moveplan"
wc -l "$moveplan"
