import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint file (e.g., ckpt/ckpt-290)")
args = parser.parse_args()

try:
    print_tensors_in_checkpoint_file(
        file_name=args.ckpt_path,
        tensor_name='', 
        all_tensors=False, 
        all_tensor_names=True
    )
except Exception as e:
    # Đã sửa cú pháp tại đây
    print("Lỗi khi đọc checkpoint: {}".format(e))