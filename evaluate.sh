#!/bin/bash

VENV_PATH="benchmark_env"
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment '$VENV_PATH' does not exist. Run setenv.py to create it and install needed packages."
    exit 1
fi

source "$VENV_PATH/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment."
    exit 1
fi

python evaluate.py "$@"

deactivate
exit 0
