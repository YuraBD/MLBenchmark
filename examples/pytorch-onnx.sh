#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

$SCRIPT_DIR/../evaluate.sh $SCRIPT_DIR/models/resnet50.pth $SCRIPT_DIR/data/resnet50_pytorch_data.npz onnx --use-softmax --model-class-path $SCRIPT_DIR/models/model_class.py --model-class-name ResNet50