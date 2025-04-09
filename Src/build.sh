#!/usr/bin/env bash
# build.sh

set -o errexit  # Exit on error

# Install system dependencies (including gfortran)
apt-get update && apt-get install -y gfortran

# Install Python dependencies
pip install -r requirements.txt