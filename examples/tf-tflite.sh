#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

$SCRIPT_DIR/../evaluate.sh $SCRIPT_DIR/models/resnet50 $SCRIPT_DIR/data/resnet50_tf_data.npz tflite --use-softmax