#!/bin/bash

# Function to create a virtual environment
create_venv() {
    echo "Creating virtual environment..."
    mkdir -p benchmark_env
    python3 -m venv benchmark_env
}

# Function to install packages
install_packages() {
    source benchmark_env/bin/activate

    local packages=()
    case $origin_fr in
        pytorch)
            packages+=("torch")
            if [[ $target_fr == "onnx" ]]; then
                packages+=("onnxruntime==1.17.1" "onnx==1.15.0")
            elif [[ $target_fr == "tflite" ]]; then
                packages+=("onnx==1.15.0" "nvidia-pyindex" "onnx-graphsurgeon" "onnxruntime==1.17.1" "onnxsim==0.4.33" "simple_onnx_processing_tools" "tensorflow==2.15.0" "protobuf==3.20.3" "onnx2tf==1.19.16" "h5py==3.7.0" "psutil==5.9.5" "ml_dtypes==0.2.0")
            fi
            ;;
        onnx)
            packages+=("onnxruntime==1.17.1")
            if [[ $target_fr == "tflite" ]]; then
                packages+=("onnx==1.15.0" "nvidia-pyindex" "onnx-graphsurgeon" "onnxsim==0.4.33" "simple_onnx_processing_tools" "tensorflow==2.15.0" "protobuf==3.20.3" "onnx2tf==1.19.16" "h5py==3.7.0" "psutil==5.9.5" "ml_dtypes==0.2.0")
            fi
            ;;
        tensorflow)
            if [[ $target_fr == "tflite" ]]; then
                packages+=("tensorflow==2.15.0")
            fi
            ;;
        tflite)
            packages+=("tensorflow==2.15.0")
            ;;
    esac

    if [[ -n "${packages[*]}" ]]; then
        echo "Installing packages: ${packages[*]}"
        pip install "${packages[@]}"
    fi

    deactivate
}

# Parse command-line arguments
origin_fr=""
target_fr=""
all=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --origin_fr) origin_fr="$2"; shift ;;
        --target_fr) target_fr="$2"; shift ;;
        --all) all=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if $all; then
    if [[ -n $origin_fr || -n $target_fr ]]; then
        echo "--all cannot be used with --origin_fr or --target_fr"
        exit 1
    fi
    origin_fr="all"
fi

if [[ -n $target_fr && -z $origin_fr ]]; then
    echo "--target_fr requires --origin_fr to be set"
    exit 1
fi

create_venv
install_packages
