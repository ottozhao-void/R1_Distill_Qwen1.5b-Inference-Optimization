#!/bin/bash

# Fix for vLLM ldconfig issue
# This script sets up the proper environment to run the inference tests
# by adding CUDA library paths and disabling problematic compilation

# Set CUDA library paths
export LD_LIBRARY_PATH="/home/zhaofanghan/tmp/lib:/home/zhaofanghan/tmp/cuda_stubs:$LD_LIBRARY_PATH"

# Add additional environment variables to help with CUDA detection
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"

echo "🔧 Environment setup for vLLM ldconfig fix"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Run the main script with passed arguments
python main.py "$@"
