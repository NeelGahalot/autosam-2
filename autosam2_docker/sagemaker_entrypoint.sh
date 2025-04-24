#!/bin/bash

# Debugging: Print environment variables
echo "=== Environment Variables ==="
env | sort

# Debugging: Print current directory and contents
echo -e "\n=== Working Directory ==="
pwd
ls -al

# Debugging: Check SageMaker specific directories
echo -e "\n=== SageMaker Directories ==="
ls -al /opt/ml 2>/dev/null || echo "/opt/ml not found"
ls -al /opt/ml/model 2>/dev/null || echo "/opt/ml/model not found"
ls -al /opt/ml/output 2>/dev/null || echo "/opt/ml/output not found"

# Run the training script with all arguments
echo -e "\n=== Starting Training Script ==="
python /app/sam2/train_sagemaker.py "$@"

# Debugging: After script runs, check what was created
#echo -e "\n=== Post-Run Directory Check ==="
#find /opt/ml -type f 2>/dev/null || echo "No files found in /opt/ml"