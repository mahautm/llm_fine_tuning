#!/usr/bin/env bash
cd /home/mmahaut/projects/paramem || exit 1

phaseplan='results/index/phase_c_artifact_plan.csv'

echo 'artifact,type,current_path,destination,action,size_mb,reason' > "$phaseplan"

# Plan for plot zips
for f in *.zip; do
  [ -f "$f" ] || continue
  size=$(du -m "$f" | cut -f1)
  if echo "$f" | grep -q 'plots_8b_layerwise'; then
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$f" "zip" "$f" "results/archive/plots_zips/$f" "archive" "$size" "plot_bundle" >> "$phaseplan"
  elif echo "$f" | grep -q 'graphs'; then
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$f" "zip" "$f" "results/archive/plots_zips/$f" "archive" "$size" "graph_bundle" >> "$phaseplan"
  else
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$f" "zip" "$f" "results/archive/$f" "archive" "$size" "other_archive" >> "$phaseplan"
  fi
done

# Plan for standalone png plots
for d in plots_*; do
  [ -d "$d" ] || continue
  
  case "$d" in
    plots_memorization)
      family="memorization"
      keep='yes'
      ;;
    plots_8b_probing)
      family="probing"
      keep='yes'
      ;;
    plots_8b_lora)
      family="lora"
      keep='yes'
      ;;
    plots_8b_layerwise*)
      family="layerwise"
      keep='yes'
      ;;
    *)
      family="other"
      keep='no'
      ;;
  esac
  
  for f in "$d"/*; do
    [ -f "$f" ] || continue
    size=$(du -m "$f" | cut -f1)
    if [ "$keep" = 'yes' ]; then
      printf '%s,%s,%s,%s,%s,%s,%s\n' "$(basename "$f")" "png" "$f" "results/final/plots/$family/$(basename "$f")" "keep" "$size" "needed_plot" >> "$phaseplan"
    else
      printf '%s,%s,%s,%s,%s,%s,%s\n' "$(basename "$f")" "png" "$f" "results/archive/old_plots/$(basename "$f")" "archive" "$size" "superseded_plot" >> "$phaseplan"
    fi
  done
done

# Root-level loose logs to be moved
for f in *.err *.out; do
  [ -f "$f" ] || continue
  size=$(du -m "$f" | cut -f1)
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$f" "log" "$f" "results/failed_runs/root_logs/$f" "move" "$size" "root_log" >> "$phaseplan"
done

echo "Phase C plan generated: $phaseplan"
wc -l "$phaseplan"
echo && echo '=== Phase C action summary ===' && awk -F, 'NR>1{c[$5]++; s[$5]+=$6} END{for (k in c) printf "%s: %d files, %d MB\n", k, c[k], s[k]}' "$phaseplan" | sort
echo && echo '=== Sample plan entries ===' && head -n 8 "$phaseplan"
