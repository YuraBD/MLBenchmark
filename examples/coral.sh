#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

$SCRIPT_DIR/../evaluate.sh $SCRIPT_DIR/models/resnet50_quantized_edgetpu.tflite $SCRIPT_DIR/data/resnet50_tf_data.npz coral --use-softmax