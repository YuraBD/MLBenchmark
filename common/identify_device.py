import os
import platform
import subprocess


def identify_device():
    device_tree_model_path = '/proc/device-tree/model'
    if os.path.exists(device_tree_model_path):
        with open(device_tree_model_path, 'r') as f:
            model_info = f.read()
            if 'Jetson Nano' in model_info:
                return "Jetson Nano"
            elif 'Raspberry Pi 4' in model_info:
                return 'Raspberry Pi 4'
            elif 'Raspberry Pi 5' in model_info:
                return 'Raspberry Pi 5'

    if platform.processor() in ('x86_64', 'AMD64'):
        return 'Desktop PC'

    return 'Unknown device'

def check_coral_usb():
    # Checking if Coral USB Accelerator is connected
    lsusb_output = subprocess.run(['lsusb'], capture_output=True, text=True).stdout
    if '1a6e:089a Global Unichip Corp.' in lsusb_output or \
       'ID 18d1:9302 Google Inc.' in lsusb_output:  # USB ID for Coral USB Accelerator
        return True
    return False


if __name__ == "__main__":
    device = identify_device()
    coral_usb = check_coral_usb()
    print(f"Current device: {device}")
    if coral_usb:
        print("Coral USB Accelerator is connected.")
    else:
        print("No Coral USB Accelerator connected.")
