#!/bin/bash


set -e

if [ -z "$1" ]; then
  read -p "Please enter the PyTorch installation directory (e.g., /path/to/pytorch): " PYTORCH_REPO
else
  PYTORCH_REPO=$1
fi

if [ -z "$2" ]; then
  read -p "Please enter the directory containing the modification files: " MOD_DIR
else
  MOD_DIR=$2
fi


if [ ! -d "$PYTORCH_REPO" ]; then
  echo "Warning: Directory $PYTORCH_REPO does not exist."
  echo "Cloning into $PYTORCH_REPO"
  git clone https://github.com/pytorch/pytorch.git "$PYTORCH_REPO"
fi

echo "Ensure that the CUDA Toolkit and cuDNN are installed and properly configured."
sleep 2

export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=1
export USE_CUSPARSELT=0
export USE_MKLDNN=1
export USE_MKL=1
export USE_DISTRIBUTED=1
export USE_OPENMP=1
export MAX_JOBS=$(nproc)
export USE_FLASH_ATTENTION=0
export PYTORCH_BUILD_VERSION=2.4.0 
export PYTORCH_BUILD_NUMBER=1 


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:/opt/conda/lib/python3.11/site-packages/torch/lib/

# --------- Check Git Repository Status ---------
cd "$PYTORCH_REPO"
git checkout v2.4.0
git clean -d -x -f
git restore .
git submodule sync
git submodule update --init --recursive

# --------- Clean Previous Builds ---------
echo "Cleaning previous build artifacts..."
python setup.py clean

# --------- Install Modifications ---------
cd ..
chmod +x install_mod.sh
./install_mod.sh $PYTORCH_REPO $MOD_DIR
cd "$PYTORCH_REPO"

# --------- Build and Install PyTorch ---------
echo "Starting PyTorch build process..."
python setup.py bdist_wheel 2>&1 | tee build_log.txt
mv dist/torch-2.4.0-cp39-cp39-linux_x86_64.whl ../

# --------- Check for Missing Libraries or Features in the Build Log ---------
echo -e "\n\nAnalyzing build log for potential issues..."
grep -E "Could NOT find|Not compiling with" build_log.txt > warnings.txt

if [ -s warnings.txt ]; then
  echo "WARNING: The following libraries or features were not found or not enabled during the build:"
  cat warnings.txt
else
  echo "No missing libraries or features detected in the build."
fi
