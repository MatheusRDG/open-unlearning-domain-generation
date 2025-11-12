#!/bin/bash
# Verify configuration files exist and check their contents

echo "=== Configuration Verification ==="
echo ""

echo "Checking for unlearn.yaml:"
if [ -f "configs/unlearn.yaml" ]; then
    echo "✓ configs/unlearn.yaml EXISTS"
    echo "File size: $(wc -c < configs/unlearn.yaml) bytes"
    echo "First 5 lines:"
    head -5 configs/unlearn.yaml
    echo ""
else
    echo "✗ configs/unlearn.yaml DOES NOT EXIST"
    echo "Files in configs/:"
    ls -la configs/*.yaml
fi

echo ""
echo "Trying to run with explicit config:"
echo "Command: uv run python src/train.py --config-path=../configs --config-name=unlearn experiment=unlearn/domain/brazil task_name=test_run --cfg job"
echo ""

uv run python src/train.py --config-path=../configs --config-name=unlearn experiment=unlearn/domain/brazil task_name=test_run --cfg job 2>&1 | head -100
