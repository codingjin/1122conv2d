import os
import csv
import time
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
import sys
import argparse
import torch
import shutil
import glob
import json
import re

def get_gpu_sm():
    """
    Returns the compute capability of the first CUDA GPU as a string, e.g., 'sm_86'.
    Returns None if no CUDA GPU is available.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    else:
        return None

target = tvm.target.cuda(arch=get_gpu_sm())

########### all pz 
sizesResnet = [
    [1, 224, 224, 64, 3, 7, 7, 2, 3],   # RESNET1
    [1, 56, 56, 64, 64, 1, 1, 1, 0],    # RESNET2
    [1, 56, 56, 64, 64, 3, 3, 1, 1],    # RESNET2
    [1, 56, 56, 256, 64, 1, 1, 1, 0],   # RESNET2
    [1, 56, 56, 128, 256, 1, 1, 2, 0],  # RESNET3
    [1, 28, 28, 128, 128, 3, 3, 1, 1],  # RESNET3
    [1, 28, 28, 512, 128, 1, 1, 1, 0],  # RESNET3
    [1, 28, 28, 256, 512, 1, 1, 2, 0],  # RESNET4
    [1, 14, 14, 256, 256, 3, 3, 1, 1],  # RESNET4
    [1, 14, 14, 1024, 256, 1, 1, 1, 0], # RESNET4
    [1, 14, 14, 512, 1024, 1, 1, 2, 0], # RESNET5
    [1, 7, 7, 512, 512, 3, 3, 1, 1],    # RESNET5
    [1, 7, 7, 2048, 512, 1, 1, 1, 0],   # RESNET5
]

sizesYolo = [
    [1, 544, 544, 32, 3, 3, 3, 1, 1],    # Yolo0
    [1, 272, 272, 64, 32, 3, 3, 1, 1],   # Yolo2
    [1, 136, 136, 128, 64, 3, 3, 1, 1],  # yolo4
    [1, 136, 136, 64, 128, 1, 1, 1, 0],  # yolo5
    [1, 68, 68, 256, 128, 3, 3, 1, 1],   # yolo8
    [1, 68, 68, 128, 256, 1, 1, 1, 0],   # yolo9
    [1, 34, 34, 512, 256, 3, 3, 1, 1],   # yolo12
    [1, 34, 34, 256, 512, 1, 1, 1, 0],   # yolo13
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],  # yolo18
    [1, 17, 17, 512, 1024, 1, 1, 1, 0],  # yolo19
]
########### all pz end

class Conv2DParams:
    def __init__(self, N, H, W, CO, CI, KH, KW, strides, padding):
        self.N = N
        self.H = H
        self.W = W
        self.CO = CO
        self.CI = CI
        self.KH = KH
        self.KW = KW
        self.strides = strides
        self.padding = padding
        
@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    """Defines the convolution workload."""
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

def calculate_conv2d_flops(N, H, W, CO, CI, KH, KW, strides, padding):
    """
    Calculate FLOPs for Conv2D operation.

    Args:
        N: Batch size
        H: Input height
        W: Input width
        CO: Output channels
        CI: Input channels
        KH: Kernel height
        KW: Kernel width
        strides: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h, pad_w)

    Returns:
        Total FLOPs
    """
    stride_h, stride_w = strides
    pad_h, pad_w = padding

    # Calculate output dimensions
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1

    # Calculate FLOPs
    # Each output element requires CI * KH * KW multiply-accumulate operations
    # Each MAC = 2 FLOPs (1 multiply + 1 add)
    flops = 2 * N * CO * OH * OW * CI * KH * KW

    return flops

def write_start_time_to_csv(log_file):
    """Writes the start time to a CSV file."""
    start_time = int(time.time())
    csv_file_path = log_file.replace('.json', '.csv')

    # write the start time to the csv file
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_file.write(f"start_time:{str(start_time)}\n")
    return start_time

def conv2d_tuning(network, specify_pz, ntrials=1000, output_dir="outputs"):
    """Tests the convolution workload with auto-scheduling."""
    print(f"ntrials: {ntrials}", flush=True)
    
    if network == "yolo":
        sizes = sizesYolo
        print(f"\ntesting yolo with {len(sizes)} layers\n")
    elif network == "resnet":
        sizes = sizesResnet
        print(f"\ntesting resnet with {len(sizes)} layers\n")
    else:
        raise Exception("network not specified!")

    # if we have specify_pz, we only test that case
    if specify_pz != -1:
        print("testing specified case: ", specify_pz, flush=True)
        if network == "yolo":
            sizes_tmp = [sizesYolo[int(specify_pz)]]
        elif network == "resnet":
            sizes_tmp = [sizesResnet[int(specify_pz)]]
        else:
            raise Exception("network not specified!")
    # otherwise, we test all cases
    else:
        print("Testing all problem sizes!", flush=True)
        sizes_tmp = sizes

    conv_params = {}
    for i, size in enumerate(sizes):
        if size not in sizes_tmp:
            continue
        N, H, W, CO, CI, KH, KW, stride, pad = size
        key = "conv" + str(i + 1)
        conv_params[key] = Conv2DParams(N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad))

    for ite, key in enumerate(conv_params.keys()):
        if specify_pz != -1:
            ite = int(specify_pz)

        conv = conv_params[key]
        #target = tvm.target.cuda(arch=get_gpu_sm())
        
        # Use the conv2d layer to test
        N, H, W, CO, CI, KH, KW, strides, padding = conv.N, conv.H, conv.W, conv.CO, conv.CI, conv.KH, conv.KW, conv.strides, conv.padding
        
        print(f"pz:{ite}, N={N}, H={H}, W={W}, CO={CO}, CI={CI}, KH={KH}, KW={KW}, strides={strides}, padding={padding}", flush=True)
        
        task = auto_scheduler.SearchTask(
            func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,
        )

        # Inspect the computational graph
        print("Computational DAG:", flush=True)
        print(task.compute_dag, flush=True)

        log_file = f"cuda_{network}_testCase_{ite}_conv2d_N_{N}_H_{H}_W_{W}_CO_{CO}_CI_{CI}_KH_{KH}_KW_{KW}_strides_{strides}_padding_{padding}.json"
        log_file = os.path.join(output_dir, log_file)
        
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ntrials,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=0,  # verbose=2
        )
        
        cost_model = auto_scheduler.XGBModel()
        search_policy = auto_scheduler.SketchPolicy(
            task, 
            program_cost_model=cost_model,
        )
        
        # skip if log_file already exists and lines of it is equal to ntrials
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == ntrials:
                    print(f"Skipping {log_file} as it already has {ntrials} trials", flush=True)
                    continue
                else:
                    os.remove(log_file)
                    csv_file_path = log_file.replace('.json', '.csv')
                    if os.path.exists(csv_file_path):
                        os.remove(csv_file_path)

        start_time = write_start_time_to_csv(log_file)

        task.tune(tune_option, search_policy)
        
        # Apply the best schedule
        try:
            sch, args = task.apply_best(log_file)

            func = tvm.build(sch, args, target)

            # Check correctness
            data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
            weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
            conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
            out_np = np.maximum(conv_np, 0.0)

            dev = tvm.cuda()
            data_tvm = tvm.nd.array(data_np, device=dev)
            weight_tvm = tvm.nd.array(weight_np, device=dev)
            out_tvm = tvm.nd.empty(out_np.shape, device=dev)
            func(data_tvm, weight_tvm, out_tvm)

            # Check results
            np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
            
            # Evaluate execution time
            evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
            print(
                f"Execution time of this operator: {np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000:.3f} ms", 
                flush=True
            )
            print(
                f"cuda testCase: {ite} conv2d for N = {N}, H = {H}, W = {W}, CO = {CO}, CI = {CI}, KH = {KH}, KW = {KW}, strides = {strides}, padding = {padding}, correctness check passed!\n", 
                flush=True
            )

        except Exception as e:
            print(f"Error during tuning or execution: {e}", flush=True)
            continue
        
        end_time = int(time.time())
        print(f"Search time: {(end_time - start_time)/60:.2f} minutes", flush=True)

def sort_json_file(input_file, output_file):
    """Sort configurations by execution time and keep only 25 representative records."""
    configs = []

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # Extract the first number from the "r" field
                if "r" in data:
                    execution_time = data["r"][0][0]
                elif "result" in data:
                    execution_time = data["result"][0][0]
                else:
                    print(f"Warning: Line {line_num} in {os.path.basename(input_file)} has no 'r' or 'result' field, skipping")
                    continue

                configs.append((execution_time, line.strip()))

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                print(f"Warning: Error parsing line {line_num} in {os.path.basename(input_file)}: {e}")
                continue

    # Sort by execution time in ascending order (fastest first)
    configs.sort(key=lambda x: x[0], reverse=False)

    # Select 25 representative records based on percentile positions
    total = len(configs)
    selected_indices = set()

    # i) Top-5 (positions 0-4)
    for i in range(5):
        if i < total:
            selected_indices.add(i)

    # ii) 10% position and next 4
    pos_10 = int(total * 0.10)
    for i in range(pos_10, min(pos_10 + 5, total)):
        selected_indices.add(i)

    # iii) 25% position and next 4
    pos_25 = int(total * 0.25)
    for i in range(pos_25, min(pos_25 + 5, total)):
        selected_indices.add(i)

    # iv) 50% position and next 4
    pos_50 = int(total * 0.50)
    for i in range(pos_50, min(pos_50 + 5, total)):
        selected_indices.add(i)

    # v) 75% position and next 4
    pos_75 = int(total * 0.75)
    for i in range(pos_75, min(pos_75 + 5, total)):
        selected_indices.add(i)

    # Write only selected configurations to output file (maintaining sorted order)
    selected_configs = [configs[i] for i in sorted(selected_indices)]

    with open(output_file, 'w') as f:
        for _, line in selected_configs:
            f.write(line + '\n')

    print(f"  Selected {len(selected_configs)} records from {total} total records")

def sort_all_results(backup_dir, output_dir):
    """Sort all JSON files from backup directory to output directory."""
    # Find all .json files in backup directory
    json_files = glob.glob(os.path.join(backup_dir, "*.json"))

    print(f"Found {len(json_files)} JSON files to sort\n")

    for json_file in json_files:
        filename = os.path.basename(json_file)
        output_file = os.path.join(output_dir, filename)

        print(f"Sorting {filename}...")
        sort_json_file(json_file, output_file)

    print(f"\nAll sorted results written to: {output_dir}")

def calculate_performance_metrics(sorted_json_dir, performance_dir):
    """Calculate GFLOPS/s for each record and save to performance directory."""

    # Create performance directory
    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    # Find all JSON files
    json_files = glob.glob(os.path.join(sorted_json_dir, "*.json"))

    print(f"Found {len(json_files)} JSON files to process\n")

    for json_file in json_files:
        filename = os.path.basename(json_file)

        # Parse parameters from filename
        # Format: cuda_{network}_testCase_{ite}_conv2d_N_{N}_H_{H}_W_{W}_CO_{CO}_CI_{CI}_KH_{KH}_KW_{KW}_strides_{strides}_padding_{padding}.json
        pattern = r'N_(\d+)_H_(\d+)_W_(\d+)_CO_(\d+)_CI_(\d+)_KH_(\d+)_KW_(\d+)_strides_\((\d+),\s*(\d+)\)_padding_\((\d+),\s*(\d+)\)'
        match = re.search(pattern, filename)

        if not match:
            print(f"Warning: Could not parse parameters from filename: {filename}")
            continue

        N, H, W, CO, CI, KH, KW, stride_h, stride_w, pad_h, pad_w = map(int, match.groups())
        strides = (stride_h, stride_w)
        padding = (pad_h, pad_w)

        # Calculate FLOPs
        flops = calculate_conv2d_flops(N, H, W, CO, CI, KH, KW, strides, padding)

        # Read JSON records and calculate performance
        performances = []
        with open(json_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # Extract execution time (in milliseconds)
                    if "r" in data:
                        execution_time_ms = data["r"][0][0]
                    elif "result" in data:
                        execution_time_ms = data["result"][0][0]
                    else:
                        continue

                    # Calculate GFLOPS/s
                    # execution_time_ms is in milliseconds, convert to seconds and then to GFLOPS/s
                    gflops_per_sec = flops / execution_time_ms / 1e6
                    performances.append(gflops_per_sec)

                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ZeroDivisionError) as e:
                    print(f"Warning: Error processing record in {filename}: {e}")
                    continue

        # Write performance data as integers
        output_file = os.path.join(performance_dir, filename)
        with open(output_file, 'w') as f:
            for perf in performances:
                f.write(f"{int(round(perf))}\n")

        print(f"Processed {filename}: {len(performances)} records, FLOPs={flops:.2e}")

    print(f"\nAll performance metrics written to: {performance_dir}")

def parse_args():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description="TVM Conv2D Auto-scheduler for YOLO and ResNet architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (both resnet and yolo, all problem sizes, 1000 trials)
  python batch_conv2d_cuda_tuning.py

  # Custom number of trials
  python batch_conv2d_cuda_tuning.py --ntrials 2000

  # Test specific problem size (e.g., layer 5)
  python batch_conv2d_cuda_tuning.py --specify_pz 5

  # Custom output directory
  python batch_conv2d_cuda_tuning.py --output_dir my_results

  # All custom options
  python batch_conv2d_cuda_tuning.py --ntrials 500 --output_dir test_results --specify_pz 3
        """
    )

    parser.add_argument(
        '--specify_pz',
        type=int,
        default=-1,
        help='Specify problem size index to test (-1 means test all, default: -1)'
    )

    parser.add_argument(
        '--ntrials',
        type=int,
        default=1000,
        help='Number of tuning trials (default: 1000)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='tuningresults',
        help='Output directory for the tuning results (default: tuningresults)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    if args.ntrials <= 0:
        raise ValueError("ntrials must be a positive integer")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Print configuration
    print("=" * 50)
    print("TVM Conv2D Auto-scheduler Configuration")
    print("=" * 50)
    print(f"Networks: ResNet and YOLO")
    print(f"Problem size index: {args.specify_pz} ({'all' if args.specify_pz == -1 else 'specific'})")
    print(f"Number of trials: {args.ntrials}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)

    # Check TVM_HOME environment variable
    tvm_home = os.environ.get("TVM_HOME")
    if tvm_home:
        print(f"TVM_HOME: {tvm_home}")
    else:
        print("Warning: TVM_HOME environment variable not set")
    print("=" * 50)

    # Run tuning for both networks
    for network in ['resnet', 'yolo']:
        print(f"\n{'='*50}")
        print(f"Starting tuning for {network.upper()} network")
        print(f"{'='*50}\n")
        conv2d_tuning(network, args.specify_pz, args.ntrials, args.output_dir)

    # Post-processing: sort all results
    print(f"\n{'='*50}")
    print("Post-processing: Sorting results")
    print(f"{'='*50}\n")

    backup_dir = args.output_dir + ".bak"

    # Rename output directory to backup
    if os.path.exists(backup_dir):
        print(f"Removing existing backup directory: {backup_dir}")
        shutil.rmtree(backup_dir)

    print(f"Renaming {args.output_dir} to {backup_dir}")
    os.rename(args.output_dir, backup_dir)

    # Create new output directory
    print(f"Creating new directory: {args.output_dir}\n")
    os.makedirs(args.output_dir)

    # Sort all JSON files
    sort_all_results(backup_dir, args.output_dir)

    print(f"\nOriginal results backed up to: {backup_dir}")
    print(f"Sorted results available in: {args.output_dir}")

    # Calculate performance metrics
    print(f"\n{'='*50}")
    print("Calculating performance metrics (GFLOPS/s)")
    print(f"{'='*50}\n")

    performance_dir = "performance"
    calculate_performance_metrics(args.output_dir, performance_dir)
