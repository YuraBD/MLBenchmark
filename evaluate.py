import argparse
from model_converted import ModelConverter
from model_evaluator import ModelEvaluator
import common.common as common


def determine_model_fr(model_path):
    while model_path[0] == '.':
        model_path = model_path[1:]
    if '.' not in model_path:
        return 'tensorflow'
    ext = model_path.split('.')[-1]
    name = model_path.split('.')[-2]
    if ext == 'pth':
        return 'pytorch'
    if ext == 'onnx':
        return 'onnx'
    if ext == 'tflite':
        if 'edgetpu' in name:
            return 'coral'
        return 'tflite'
    if ext == 'trt':
        return 'tensorrt'
    else:
        raise Exception(f'Unsupported model type - {ext}')


def get_input_shape(data_path):
    x_data, _ = common.load_data(data_path)
    input_shape = list(x_data.shape)
    input_shape[0] = 1
    return input_shape


def runner_init(args):
    if args.framework != 'tflite' and args.use_armnn:
        raise Exception('ARMNN is currenly supported only for tflite inference')
    RunnerClass = common.get_runner_class(args.framework)
    if args.use_armnn and args.framework == 'tflite':
        return RunnerClass(args.model_path, args.armnn_path, {"backends": "CpuAcc,CpuRef", "logging-severity":"warning"})
    if args.quantization and args.framework == 'tflite':
        return RunnerClass(args.model_path, quantization = args.quantization)
    if args.framework == 'pytorch':
        return RunnerClass(args.model_path, args.model_class_path, args.model_class_name)
    return RunnerClass(args.model_path)


def evaluate(args):
    print('Evaluating model')
    print(f'Model path:        {args.model_path}')
    print(f'Data path:         {args.data_path}')
    print(f'Framework:         {args.framework}')
    if args.framework == 'pytorch':
        print(f'Model class path:  {args.model_class_path}')
        print(f'Model class name:  {args.model_class_name}')
    if args.framework == 'tflite':
        if args.use_armnn:
            print(f'Use delegate:      {args.use_armnn}')
            print(f'Delegate path:     {args.armnn_path}')

    runner = runner_init(args)
    model_evaluator = ModelEvaluator(runner, args.data_path, args.model_type, use_softmax = args.use_softmax)
    metrics = model_evaluator.evaluate_model()
    print(f'Average inference time : {metrics["aver_inf_time"]} seconds')
    if args.model_type == 'classification':
        print(f'Accuracy -             : {metrics["accuracy"]} %')
        print(f'Precision              : {metrics["precision"]}')
        print(f'Recall                 : {metrics["recall"]}')
        print(f'F1 Score               : {metrics["f1"]}')
    else:
        print(f'MAPE                   : {metrics["mape"]} %')
        print(f'MSE                    : {metrics["mse"]}')



def parse_args():
    parser = argparse.ArgumentParser(description='Process some inputs for model inference.')

    parser.add_argument('model_path', type=str, help='Path to the model file')
    parser.add_argument('data_path', type=str, help='Path to the data file')
    parser.add_argument('framework', type=str, choices=['pytorch', 'tflite', 'onnx', 'tensorrt', 'coral'],
                        help='The framework of the model (e.g., pytorch, tensorflow, onnx)')
    parser.add_argument('--use-armnn', action='store_true', help='Use ARMNN delegate to speed up inference on ARM architecture CPU')
    parser.add_argument('--armnn-path', type=str, default='ArmNN-aarch64/libarmnnDelegate.so')
    parser.add_argument('--use-softmax', action='store_true')
    parser.add_argument('--model-type', type=str, default='classification', choices=['classification', 'regression'],
                        help='Type of the model (classification or regression)')
    parser.add_argument('--model-class-path', type=str, default=None,
                        help='The path to the model class file (optional)')
    parser.add_argument('--model-class-name', type=str, default=None,
                        help='The name of the model class (optional)')
    parser.add_argument('--do-not-keep-shape', action='store_true')
    parser.add_argument('--transposed-data-path', type=str, help='Path to the transposed data file')
    parser.add_argument('--convert-only', action='store_true')
    parser.add_argument('--quantization', action='store_true')
    parser.add_argument('--representative-data-path', type=str, default=None)

    args = parser.parse_args()

    model_fr = determine_model_fr(args.model_path)
    if model_fr != 'tensorflow' and args.framework == 'tflite' and args.do_not_keep_shape:
        if not args.transposed_data_path:
            raise Exception('If you do not want to use transpose in converted tflite model, provide transposed data: --transpose-data-path path')

    if args.quantization and not args.representative_data_path and args.framework != 'tflite':
        raise Exception('For full-integer quantization representative dataset should be provided: --representative-data-path path ')

    return args


def main():
    args = parse_args()
    model_fr = determine_model_fr(args.model_path)
    if model_fr != args.framework:
        if args.framework == 'coral':
            raise Exception('Conversion to coral must be done manually. Refer to README.')
        if model_fr != 'tensorflow' and args.framework == 'tflite' and args.do_not_keep_shape:
            if not args.transposed_data_path:
                raise Exception('If you do not want to use transpose in converted tflite model, provide transposed data in --transpose-data-path')
        print(f'Converting model {args.model_path} from {model_fr} to {args.framework}')
        model_converter = ModelConverter(args.model_path,
                                         args.do_not_keep_shape,
                                         quantization = args.quantization,
                                         model_class_path=args.model_class_path,
                                         model_class_name=args.model_class_name,
                                         input_shape=get_input_shape(args.data_path),
                                         model_type=args.model_type,
                                         representative_data_path=args.representative_data_path)
        args.model_path = model_converter.convert(model_fr, args.framework)
        if model_fr != 'tensorflow' and args.framework == 'tflite' and args.do_not_keep_shape:
            args.data_path = args.transposed_data_path
    if not args.convert_only:
        evaluate(args)


if __name__ == '__main__':
    main()