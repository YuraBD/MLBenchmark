from inference_runner import InferenceRunner
import argparse
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(inf_runner: InferenceRunner):
    inf_times, correct_pred = inf_runner.run_inference()
    data_len = inf_runner.get_data_len()

    print(f"Average inference time per image - {sum(inf_times)/data_len} sec")
    print(f"Accuracy - {(correct_pred / data_len) * 100}%")

def main():
    parser = argparse.ArgumentParser(description='Process some inputs for model inference.')

    parser.add_argument('model_path', type=str, help='Path to the model file')
    parser.add_argument('data_path', type=str, help='Path to the data file')
    parser.add_argument('framework', type=str, choices=['pytorch', 'tflite', 'onnx', 'tensorrt'],
                        help='The framework of the model (e.g., pytorch, tensorflow, onnx)')
    parser.add_argument('--model_type', type=str, default='classification', choices=['classification', 'regression'],
                        help='Type of the model (classification or regression)')
    parser.add_argument('--model_class_path', type=str, default=None,
                        help='The path to the model class file (optional)')
    parser.add_argument('--model_class_name', type=str, default=None,
                        help='The name of the model class (optional)')

    args = parser.parse_args()
    print("------------------------------", args.model_path)
    inf_runner = InferenceRunner(args.model_path, args.data_path, args.framework, args.model_type, args.model_class_path, args.model_class_name)

    evaluate_model(inf_runner)


if __name__ == "__main__":
    main()
