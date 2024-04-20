import argparse
from model_converted import ModelConverter
from model_evaluator import ModelEvaluator
import common.common as common


def determine_model_fr(model_path):
    if '.' not in model_path:
        return 'tensorflow'
    ext = model_path.split('.')[-1]
    if ext == 'pth':
        return 'pytorch'
    if ext == 'onnx':
        return 'onnx'
    if ext == 'tflite':
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
    if args.use_armnn:
        return RunnerClass(args.model_path, args.armnn_dir, {"backends": "CpuAcc,CpuRef", "logging-severity":"warning"})
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
            print(f'Delegate path:     {args.armnn_dir}')

    runner = runner_init(args)
    model_evaluator = ModelEvaluator(runner, args.data_path, args.model_type, args.use_softmax)
    accuracy, aver_inf_time = model_evaluator.evaluate_model()
    print(f'Accuracy - {accuracy} %')
    print(f'Average inference time - {aver_inf_time} seconds')


def parse_args():
    parser = argparse.ArgumentParser(description='Process some inputs for model inference.')

    parser.add_argument('model_path', type=str, help='Path to the model file')
    parser.add_argument('data_path', type=str, help='Path to the data file')
    parser.add_argument('framework', type=str, choices=['pytorch', 'tflite', 'onnx', 'tensorrt'],
                        help='The framework of the model (e.g., pytorch, tensorflow, onnx)')
    parser.add_argument('--use-armnn', action='store_true', help='Use ARMNN delegate to speed up inference on ARM architecture CPU')
    parser.add_argument('--armnn-dir', type=str, default='ArmNN-aarch64')
    parser.add_argument('--use-softmax', action='store_true')
    parser.add_argument('--model-type', type=str, default='classification', choices=['classification', 'regression'],
                        help='Type of the model (classification or regression)')
    parser.add_argument('--model-class-path', type=str, default=None,
                        help='The path to the model class file (optional)')
    parser.add_argument('--model-class-name', type=str, default=None,
                        help='The name of the model class (optional)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_fr = determine_model_fr(args.model_path)
    if model_fr != args.framework:
        print(f'Converting model {args.model_path} from {model_fr} to {args.framework}')
        model_converter = ModelConverter(args.model_path,
                                         model_class_path=args.model_class_path,
                                         model_class_name=args.model_class_name,
                                         input_shape=get_input_shape(args.data_path),
                                         data_path=args.data_path)
        args.model_path, conv_data_path = model_converter.convert(model_fr, args.framework)
        if args.data_path != conv_data_path:
            args.data_path = conv_data_path

    evaluate(args)


if __name__ == '__main__':
    main()