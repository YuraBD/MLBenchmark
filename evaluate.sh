#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

VENV_PATH="$SCRIPT_DIR/benchmark_env"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment '$VENV_PATH' does not exist. Run setenv.py to create it and install needed packages."
    exit 1
fi

source "$VENV_PATH/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

if [[ " $@ " =~ " --use-armnn " ]]; then
    ARMNN_PATH_ARG=$(echo "$@" | grep -oP '(?<=--armnn-path )[^ ]+')
    if [ -z "$ARMNN_PATH_ARG" ]; then
        # --armnn-path not set, use default path
        export LD_LIBRARY_PATH="$SCRIPT_DIR/ArmNN-aarch64:${LD_LIBRARY_PATH}"
    else
        # --armnn-path is set, extract directory and add to LD_LIBRARY_PATH
        ARMNN_DIR=$(dirname "$(realpath "$ARMNN_PATH_ARG")")
        export LD_LIBRARY_PATH="${ARMNN_DIR}:${LD_LIBRARY_PATH}"
    fi
fi

python $SCRIPT_DIR/evaluate.py "$@"

deactivate
exit 0
