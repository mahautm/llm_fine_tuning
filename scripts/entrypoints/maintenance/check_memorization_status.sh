#!/bin/bash
# check_memorization_status.sh
# Quick status check for memorization evaluation batch jobs

echo "=========================================="
echo "Memorization Evaluation Status"
echo "=========================================="
echo ""

# Count jobs in queue
RUNNING=$(squeue -u $USER | grep "mem_" | grep " R " | wc -l)
PENDING=$(squeue -u $USER | grep "mem_" | grep " PD " | wc -l)
echo "🔄 Jobs in Queue:"
echo "   Running: $RUNNING"
echo "   Pending: $PENDING"
echo ""

# Count completed evaluations
TOTAL_CHECKPOINTS=$(find /home/mmahaut/projects/paramem/models3 -type d -name "slurm_logs" | wc -l)
COMPLETED=$(find /home/mmahaut/projects/paramem/models3 -name "memorization_metrics.json" 2>/dev/null | wc -l)
COMPLETED_OUT=$(grep -l "COMPLETED" /home/mmahaut/projects/paramem/models3/*/checkpoint-*/slurm_logs/memorization_metrics.out 2>/dev/null | wc -l)

echo "📊 Evaluation Progress:"
echo "   Total checkpoints with slurm_logs: $TOTAL_CHECKPOINTS"
echo "   Completed (JSON exists): $COMPLETED"
echo "   Completed (COMPLETED marker): $COMPLETED_OUT"
echo "   Progress: $(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_CHECKPOINTS)*100}")%"
echo ""

# Show by model
echo "📁 By Model:"
for MODEL_DIR in /home/mmahaut/projects/paramem/models3/Llama-3.1-8B*/; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    TOTAL=$(find "$MODEL_DIR" -type d -name "slurm_logs" | wc -l)
    DONE=$(find "$MODEL_DIR" -name "memorization_metrics.json" 2>/dev/null | wc -l)
    echo "   $MODEL_NAME: $DONE/$TOTAL"
done
echo ""

# Check for recent errors
ERRORS=$(find /home/mmahaut/projects/paramem/models3 -name "memorization_metrics.err" -mmin -60 -size +0 2>/dev/null)
if [ -n "$ERRORS" ]; then
    echo "⚠️  Recent Errors (last hour):"
    echo "$ERRORS" | while read errfile; do
        echo "   $(dirname $errfile | sed 's|.*/models3/||')"
    done
    echo ""
fi

# Show latest completions
echo "✅ Latest Completions:"
find /home/mmahaut/projects/paramem/models3 -name "memorization_metrics.out" -mmin -120 2>/dev/null | \
    xargs grep -l "COMPLETED" 2>/dev/null | \
    sed 's|.*/models3/||' | sed 's|/slurm_logs.*||' | \
    tail -5 | while read line; do
        echo "   $line"
    done
echo ""

echo "=========================================="
echo "Commands:"
echo "  Monitor jobs:  squeue -u \$USER | grep mem_"
echo "  Cancel all:    scancel -u \$USER -n mem_"
echo "  Analyze:       srun --ntasks=1 python scripts/analysis/analyze_memorization_all_checkpoints.py"
echo "=========================================="
