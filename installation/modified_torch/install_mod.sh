#!/bin/bash

if [ -z "$1" ]; then
  read -p "Please enter the PyTorch installation directory (e.g., /path/to/pytorch): " TORCH_DIR
else
  TORCH_DIR=$1
fi

if [ -z "$2" ]; then
  read -p "Please enter the directory containing the modification files: " MOD_DIR
else
  MOD_DIR=$2
fi

if [ ! -d "$TORCH_DIR" ]; then
  echo "Error: Directory $TORCH_DIR does not exist."
  exit 1
fi

if [ ! -d "$MOD_DIR" ]; then
  echo "Error: Directory $MOD_DIR does not exist."
  exit 1
fi

FILES=("TensorImpl.cpp" "TensorImpl.h" "Module.cpp" "__init__.py" "python_variable.cpp" "select_compute_arch.cmake")

DEST_PATHS=(
  "$TORCH_DIR/c10/core/TensorImpl.cpp"
  "$TORCH_DIR/c10/core/TensorImpl.h"
  "$TORCH_DIR/torch/csrc/Module.cpp"
  "$TORCH_DIR/torch/__init__.py"
  "$TORCH_DIR/torch/csrc/autograd/python_variable.cpp"
  "$TORCH_DIR/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
)

for i in ${!FILES[@]}; do
  SRC_FILE="$MOD_DIR/${FILES[$i]}"
  DEST_FILE="${DEST_PATHS[$i]}"

  if [ ! -f "$SRC_FILE" ]; then
    echo "Error: Source file $SRC_FILE does not exist in $MOD_DIR."
    exit 1
  fi

  cp -vb "$SRC_FILE" "$DEST_FILE"

  if [ $? -eq 0 ]; then
    echo "$SRC_FILE copied to $DEST_FILE successfully."
  else
    echo "Error copying $SRC_FILE to $DEST_FILE."
    exit 1
  fi
done

echo "All files copied successfully."
