#!/bin/bash
# Debug script to check configuration files

echo "=== Checking Configuration Files ==="
echo ""

echo "1. Checking experiment config:"
if [ -f "configs/experiment/unlearn/domain/brazil.yaml" ]; then
    echo "✓ configs/experiment/unlearn/domain/brazil.yaml EXISTS"
    echo "Content:"
    cat configs/experiment/unlearn/domain/brazil.yaml
else
    echo "✗ configs/experiment/unlearn/domain/brazil.yaml MISSING"
fi
echo ""

echo "2. Checking dataset configs:"
if [ -f "configs/data/datasets/DOMAIN_brazil_forget.yaml" ]; then
    echo "✓ configs/data/datasets/DOMAIN_brazil_forget.yaml EXISTS"
    cat configs/data/datasets/DOMAIN_brazil_forget.yaml
else
    echo "✗ configs/data/datasets/DOMAIN_brazil_forget.yaml MISSING"
fi
echo ""

if [ -f "configs/data/datasets/DOMAIN_brazil_retain.yaml" ]; then
    echo "✓ configs/data/datasets/DOMAIN_brazil_retain.yaml EXISTS"
    cat configs/data/datasets/DOMAIN_brazil_retain.yaml
else
    echo "✗ configs/data/datasets/DOMAIN_brazil_retain.yaml MISSING"
fi
echo ""

echo "3. Checking data directories:"
ls -la data/run/ 2>&1
echo ""

echo "4. Testing Hydra config loading:"
uv run python src/train.py --config-name=unlearn \
    experiment=unlearn/domain/brazil \
    task_name=test_config_check \
    --cfg job 2>&1 | head -50
