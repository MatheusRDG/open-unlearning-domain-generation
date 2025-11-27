#!/bin/bash
# Clean up cached/saved experiments
#
# Usage:
#   bash scripts/clean-experiments.sh              # Interactive mode (shows what will be deleted)
#   bash scripts/clean-experiments.sh --all        # Delete everything
#   bash scripts/clean-experiments.sh --models     # Delete only model checkpoints
#   bash scripts/clean-experiments.sh --data       # Delete only generated data
#   bash scripts/clean-experiments.sh --results    # Delete only evaluation results
#   bash scripts/clean-experiments.sh --configs    # Delete only generated configs
#   bash scripts/clean-experiments.sh --dry-run    # Show what would be deleted without deleting

set -e

# Parse arguments
DRY_RUN=false
DELETE_ALL=false
DELETE_MODELS=false
DELETE_DATA=false
DELETE_RESULTS=false
DELETE_CONFIGS=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --all)
            DELETE_ALL=true
            ;;
        --models)
            DELETE_MODELS=true
            ;;
        --data)
            DELETE_DATA=true
            ;;
        --results)
            DELETE_RESULTS=true
            ;;
        --configs)
            DELETE_CONFIGS=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: bash scripts/clean-experiments.sh [--all|--models|--data|--results|--configs] [--dry-run]"
            exit 1
            ;;
    esac
done

# If no specific flags, run interactive mode
if [ "$DELETE_ALL" = false ] && [ "$DELETE_MODELS" = false ] && [ "$DELETE_DATA" = false ] && [ "$DELETE_RESULTS" = false ] && [ "$DELETE_CONFIGS" = false ]; then
    INTERACTIVE=true
else
    INTERACTIVE=false
fi

echo "============================================"
echo "Experiment Cleanup Script"
echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "MODE: Dry run (no files will be deleted)"
fi
echo ""

# Function to show size and delete
cleanup_dir() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ] || [ -L "$dir" ]; then
        # Handle symlinks
        if [ -L "$dir" ]; then
            real_path=$(readlink -f "$dir")
            size=$(du -sh "$real_path" 2>/dev/null | cut -f1 || echo "0")
            echo "  $name: $size (symlink -> $real_path)"
        else
            size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "0")
            echo "  $name: $size"
        fi

        if [ "$DRY_RUN" = false ]; then
            rm -rf "$dir"
            echo "    ✓ Deleted"
        else
            echo "    [dry-run] Would delete"
        fi
    else
        echo "  $name: Not found"
    fi
}

# Model checkpoints
if [ "$DELETE_ALL" = true ] || [ "$DELETE_MODELS" = true ] || [ "$INTERACTIVE" = true ]; then
    echo "MODEL CHECKPOINTS:"
    if [ "$INTERACTIVE" = true ]; then
        read -p "  Delete model checkpoints? (y/N) " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            DELETE_MODELS=true
        fi
    fi

    if [ "$DELETE_MODELS" = true ] || [ "$DELETE_ALL" = true ]; then
        cleanup_dir "./saves/train" "Finetuned models (saves/train)"
        cleanup_dir "./saves/unlearn" "Unlearned models (saves/unlearn)"
        cleanup_dir "./saves/eval" "Eval outputs (saves/eval)"
    fi
    echo ""
fi

# Generated data
if [ "$DELETE_ALL" = true ] || [ "$DELETE_DATA" = true ] || [ "$INTERACTIVE" = true ]; then
    echo "GENERATED DATA:"
    if [ "$INTERACTIVE" = true ]; then
        read -p "  Delete generated datasets? (y/N) " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            DELETE_DATA=true
        fi
    fi

    if [ "$DELETE_DATA" = true ] || [ "$DELETE_ALL" = true ]; then
        cleanup_dir "./data/run" "Run data (data/run)"
        cleanup_dir "./output" "Domain JSON outputs (output)"
        cleanup_dir "./.logs/generations" "Generation checkpoints (.logs/generations)"
    fi
    echo ""
fi

# Results
if [ "$DELETE_ALL" = true ] || [ "$DELETE_RESULTS" = true ] || [ "$INTERACTIVE" = true ]; then
    echo "EVALUATION RESULTS:"
    if [ "$INTERACTIVE" = true ]; then
        read -p "  Delete evaluation results? (y/N) " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            DELETE_RESULTS=true
        fi
    fi

    if [ "$DELETE_RESULTS" = true ] || [ "$DELETE_ALL" = true ]; then
        cleanup_dir "./results" "Evaluation CSVs (results)"
    fi
    echo ""
fi

# Generated configs
if [ "$DELETE_ALL" = true ] || [ "$DELETE_CONFIGS" = true ] || [ "$INTERACTIVE" = true ]; then
    echo "GENERATED CONFIGS:"
    if [ "$INTERACTIVE" = true ]; then
        read -p "  Delete generated configs? (y/N) " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            DELETE_CONFIGS=true
        fi
    fi

    if [ "$DELETE_CONFIGS" = true ] || [ "$DELETE_ALL" = true ]; then
        # Only delete DOMAIN_* configs, not base configs
        if [ -d "./configs/data/datasets" ]; then
            echo "  Domain dataset configs:"
            for f in ./configs/data/datasets/DOMAIN_*.yaml; do
                if [ -f "$f" ]; then
                    echo "    $f"
                    if [ "$DRY_RUN" = false ]; then
                        rm -f "$f"
                    fi
                fi
            done
        fi

        cleanup_dir "./configs/experiment/finetune/domain" "Finetune experiment configs"
        cleanup_dir "./configs/experiment/unlearn/domain" "Unlearn experiment configs"
    fi
    echo ""
fi

echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No files were deleted."
    echo "Remove --dry-run to actually delete files."
else
    echo "Cleanup complete!"
fi
echo "============================================"
