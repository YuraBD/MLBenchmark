import argparse
import os
import subprocess
import common.identify_device as ident_dev


def parse_args():
    parser = argparse.ArgumentParser(description='Usage of setenv.py')
    parser.add_argument('--origin-fr', type=str, choices=['pytorch', 'onnx', 'tensorflow', 'tflite', 'tensorrt', 'coral'],
                        help='Specify only this if you already have converted model and want just to evaluate it')
    parser.add_argument('--target-fr', type=str, choices=['onnx', 'tflite', 'tensorrt', 'coral'],
                        help='Specify this and --origin-fr if you want to install dependencies that will be needed to convert\
                            model from --origin-fr to --target-fr and measure performance')
    parser.add_argument('--all', action='store_true',
                        help='Install all dependencies')
    
    args = parser.parse_args()

    if args.all:
        if args.origin_fr or args.target_fr:
            raise ValueError('--all cannot be used with --origin_fr or --target_fr')
    elif args.target_fr and not args.origin_fr:
        raise ValueError('--target_fr requires --origin_fr to be set')

    return args

def create_venv():
    print('Creating virtual environment...')
    os.makedirs('benchmark_env', exist_ok=True)
    subprocess.run(['python3.9', '-m', 'venv', 'benchmark_env'])

def install_packages(args):
    activate_script = os.path.join('benchmark_env', 'bin', 'activate')

    if args.all or not args.origin_fr and not args.target_fr:
        packages = [
            'scikit-learn',
            'torch',
            'torchvision',
            'onnx==1.15.0',
            'nvidia-pyindex',
            'onnx-graphsurgeon',
            'onnxruntime==1.17.1',
            'onnxsim==0.4.33',
            'simple_onnx_processing_tools',
            'flatbuffers==24.3.25',
            'tensorflow==2.15.0',
            'protobuf==3.20.3',
            'onnx2tf==1.19.16',
            'h5py==3.7.0',
            'psutil==5.9.5',
            'ml_dtypes==0.2.0'
        ]
        print('Installing all packages...')
    else:
        packages = ['scikit-learn', 'numpy']
        if args.origin_fr == 'pytorch':
            packages.extend(['torch', 'torchvision'])
            if args.target_fr == 'onnx':
                packages.extend(['onnxruntime==1.17.1', 'onnx==1.15.0'])
            elif args.target_fr in ['tflite', 'coral']:
                packages.extend([
                    'onnx==1.15.0', 'flatbuffers==24.3.25', 'nvidia-pyindex', 'onnx-graphsurgeon', 'onnxruntime==1.17.1',
                    'onnxsim==0.4.33', 'simple_onnx_processing_tools', 'tensorflow==2.15.0',
                    'protobuf==3.20.3', 'onnx2tf==1.19.16', 'h5py==3.7.0', 'psutil==5.9.5', 'ml_dtypes==0.2.0'
                ])
        elif args.origin_fr == 'onnx':
            packages.append('onnxruntime==1.17.1')
            if args.target_fr in ['tflite', 'coral']:
                packages.extend([
                    'onnx==1.15.0', 'flatbuffers==24.3.25', 'nvidia-pyindex', 'onnx-graphsurgeon',
                    'onnxsim==0.4.33', 'simple_onnx_processing_tools', 'tensorflow==2.15.0',
                    'protobuf==3.20.3', 'onnx2tf==1.19.16', 'h5py==3.7.0', 'psutil==5.9.5', 'ml_dtypes==0.2.0'
                ])
        elif args.origin_fr == 'tensorflow' or args.origin_fr == 'tflite':
            packages.extend(['flatbuffers==24.3.25', 'tensorflow==2.15.0'])
        
        if args.target_fr == 'tensorrt':
            print('TensorRT must be installed manually. Dependencies are described in the README.')

    # Construct the command to activate and install packages
    command = f'source {activate_script} && pip install --upgrade pip && pip install ' + ' '.join(packages)
    subprocess.run(command, shell=True, executable='/bin/bash')
    if not args.origin_fr or args.origin_fr == 'coral' or args.target_fr == 'coral':
        command = f'source {activate_script} && pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0'
        subprocess.run(command, shell=True, executable='/bin/bash')


def recommend_config():
    device = ident_dev.identify_device()
    if device == 'Desktop PC':
        print('You are using desktop.')
    elif device == 'Jetson Nano':
        print('You are using Jetson Nano.')
        print('To achive the best performance it is recommended to use tensorrt to achive the fastest inference.')
    elif device in ['Raspberry Pi 4', 'Raspberry Pi 5']:
        print(f'You are using {device}.')
        coral_connected = ident_dev.check_coral_usb()
        if coral_connected:
            print('To achive the best performance it is recommended to use TensorFlow Lite as Coral USB Accelerator is connected.')
        else:
            print('To achive the best performance it is recomended to use tflite with ARMNN delegate')


if __name__ == '__main__':
    args = parse_args()
    print('Creating venv')
    create_venv()
    print('Installing packages')
    install_packages(args)
    print('All packages have been installed successfully')
    recommend_config()
