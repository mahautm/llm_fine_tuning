#!/usr/bin/env bash
cd /home/mmahaut/projects/paramem || exit 1

manifest='results/index/run_manifest.csv'
failcsv='results/index/failure_signatures.csv'

echo 'run_id,family,mode,dataset,status,reason,has_wandb,output_path,log_err,log_out,kept_as' > "$manifest"
echo 'run_id,source,signature,log_err' > "$failcsv"

classify_file() {
  local f="$1"
  local source="$2"
  local base name run_id family mode dataset status reason has_wandb outpair out_guess

  base=$(basename "$f")
  name="${base%.err}"

  run_id='unknown'
  if echo "$name" | grep -Eq '_[0-9]{6,}$'; then
    run_id=$(echo "$name" | sed -E 's/.*_([0-9]{6,})$/\1/')
  fi

  family='other'
  if echo "$name" | grep -qi 'layerwise'; then family='layerwise'; fi
  if echo "$name" | grep -qi 'probing'; then family='probing'; fi
  if echo "$name" | grep -qi 'memorization|mem_'; then family='memorization'; fi
  if echo "$name" | grep -qi 'nccl'; then family='infra'; fi
  if echo "$name" | grep -qi 'analysis|current_analysis'; then family='analysis'; fi

  mode='analysis'
  if echo "$name" | grep -qi 'lora'; then mode='lora'; fi
  if echo "$name" | grep -qi 'full'; then mode='full'; fi
  if echo "$name" | grep -qi 'test_7b_model'; then mode='baseline'; fi
  if echo "$name" | grep -qi 'nccl'; then mode='infra'; fi

  dataset='unknown'
  if echo "$name" | grep -qi 'wikiplus'; then dataset='wikiplus'; fi
  if echo "$name" | grep -qi 'pile'; then dataset='pile'; fi
  if echo "$name" | grep -qi 'mmlu'; then dataset='mmlu'; fi

  has_wandb='no'
  if grep -Eq 'wandb: Synced|View run .*wandb.ai|View run /home/.*/runs/' "$f"; then
    has_wandb='yes'
  fi

  status='unknown'
  reason='no-clear-terminal-signal'

  if grep -Eqi 'oom-kill|out of memory|cuda out of memory|SIGKILL' "$f"; then
    status='failed'; reason='OOM'
  elif grep -Eqi 'TIME LIMIT|DUE TO TIME LIMIT|CANCELLED AT .* DUE TO TIME LIMIT' "$f"; then
    status='failed'; reason='TIME_LIMIT'
  elif grep -Eqi "can't open file|No such file or directory|execve\(\):" "$f"; then
    status='failed'; reason='PATH_ENV'
  elif grep -Eqi 'traceback \(most recent call last\)|exception|runtimeerror|keyerror|\[rank[0-9]+\].*traceback' "$f"; then
    status='failed'; reason='TRACEBACK'
  elif [ "$has_wandb" = 'yes' ] && ! grep -Eqi 'traceback \(most recent call last\)|oom-kill|out of memory|TIME LIMIT|DUE TO TIME LIMIT|No such file or directory|execve\(\):' "$f"; then
    status='success'; reason='wandb-synced-no-terminal-failure'
  fi

  outpair="${f%.err}.out"
  out_guess=''
  if [ -f "$outpair" ]; then out_guess="$outpair"; fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$family" "$mode" "$dataset" "$status" "$reason" "$has_wandb" "" "$f" "$out_guess" "review" >> "$manifest"

  if grep -Eqi 'oom-kill|out of memory|cuda out of memory|SIGKILL' "$f"; then
    printf '%s,%s,%s,%s\n' "$run_id" "$source" 'OOM' "$f" >> "$failcsv"
  fi
  if grep -Eqi 'TIME LIMIT|DUE TO TIME LIMIT|CANCELLED AT .* DUE TO TIME LIMIT' "$f"; then
    printf '%s,%s,%s,%s\n' "$run_id" "$source" 'TIME_LIMIT' "$f" >> "$failcsv"
  fi
  if grep -Eqi "can't open file|No such file or directory|execve\(\):|commandnotfounderror|module\(s\) are unknown" "$f"; then
    printf '%s,%s,%s,%s\n' "$run_id" "$source" 'PATH_ENV' "$f" >> "$failcsv"
  fi
  if grep -Eqi 'traceback \(most recent call last\)|exception|runtimeerror|keyerror|\[rank[0-9]+\].*traceback' "$f"; then
    printf '%s,%s,%s,%s\n' "$run_id" "$source" 'TRACEBACK' "$f" >> "$failcsv"
  fi
  if grep -Eqi 'NCCL|ProcessGroupNCCL|destroy_process_group' "$f"; then
    printf '%s,%s,%s,%s\n' "$run_id" "$source" 'NCCL' "$f" >> "$failcsv"
  fi
  if grep -Eqi 'commandnotfounderror|module\(s\) are unknown|cuda/[0-9]+' "$f"; then
    printf '%s,%s,%s,%s\n' "$run_id" "$source" 'ENV' "$f" >> "$failcsv"
  fi
}

for f in slurm_logs/*.err; do
  [ -f "$f" ] && classify_file "$f" 'slurm_logs'
done

for f in *.err; do
  [ -f "$f" ] && classify_file "$f" 'root'
done

echo "manifest_lines=$(wc -l < "$manifest")"
echo "failure_lines=$(wc -l < "$failcsv")"
