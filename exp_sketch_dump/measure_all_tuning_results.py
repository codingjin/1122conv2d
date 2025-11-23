import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.auto_scheduler.measure_record import load_record_from_string
import json
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
import os
import glob
import csv
import re
import argparse

def get_verify_pass(valid, **kwargs):
    print(kwargs)
    def _fverify(f, *_):
        print(f)
        valid[0] = tvm.tir.analysis.verify_gpu_code(f, kwargs)
        return f

    return tvm.tir.transform.prim_func_pass(_fverify, opt_level=0)

def extract_values_from_json(line):
    # print(f"line: {line}")
    data = json.loads(line)
    value_str = data['i'][0][0]
    value_list = json.loads(value_str)
    pz = value_list[1:]
    # print(f"pz: {pz}")
    return pz

def extract_layer_name_from_filename(filename):
    """Extract layer identifier from tuning result filename."""
    # Example: cuda_resnet_testCase_0_conv2d_N_1_H_224_W_224...
    # Extract: cuda_resnet_testCase_0
    basename = os.path.basename(filename)
    match = re.match(r'(cuda_(resnet|yolo)_testCase_\d+)_', basename)
    if match:
        return match.group(1)
    return basename.replace('.json', '')

def extract_short_layer_name(filename):
    """Extract short layer name for display."""
    # Example: cuda_resnet_testCase_0 -> resnet_layer_0
    basename = os.path.basename(filename)
    match = re.match(r'cuda_(resnet|yolo)_testCase_(\d+)_', basename)
    if match:
        network = match.group(1)
        testcase = match.group(2)
        return f"{network}_layer_{testcase}"
    return basename.replace('.json', '')

@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

target = tvm.target.Target("cuda")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate CUDA kernels from TVM tuning results and measure energy consumption')
args = parser.parse_args()

# Fixed measurement parameters (no test/full mode distinction)
num_lrounds = 30
num_rounds = 100
total_executions = num_lrounds * num_rounds * 1000

print("\n" + "="*80)
print("  ENERGY MEASUREMENT CONFIGURATION")
print(f"  Large rounds (lrounds): {num_lrounds}")
print(f"  Target executions per lround: 100,000 (adaptive: iteration × round)")
print(f"  Total target executions per kernel: {total_executions:,}")
print(f"  Note: iteration ∈ {{1000, 100, 10}} adjusted based on GPU memory")
print("="*80 + "\n")

# Find all JSON files in ../tuningresults/ directory
tuning_results_dir = "../tuningresults"
if not os.path.exists(tuning_results_dir):
    print(f"Error: Directory {tuning_results_dir} not found!")
    print("Make sure you run this script from the exp_sketch_dump/ directory")
    exit(1)

json_files = sorted(glob.glob(os.path.join(tuning_results_dir, "*.json")))
print(f"Found {len(json_files)} JSON files in {tuning_results_dir}/")

if len(json_files) == 0:
    print("Error: No JSON files found in tuningresults directory!")
    exit(1)

str_headers = '''
#include <cassert>
#include <stdlib.h>
#include <cuda.h>

'''

class RecordProcessor:
    IDX_NODE_NAME = 0
    IDX_STAGE = 1
    IDX_ITER = 2
    IDX_LOOP_EXTENT = 3
    IDX_LENGTHS = 4
    IDX_INNER_TO_OUTER = 5
    IDX_TASK = 0
    IDX_STATE = 1
    IDX_TB = 2
    LENGTH_PAR_DIM = 4
    LENGTH_REDUC = 2

    def __init__(self, record):
        self.record = record
        self.json_str = json.loads(record)

file_path = "demo.cu"

# Create main directories
os.makedirs("kernels", exist_ok=True)
os.makedirs("kernel_outputs", exist_ok=True)
os.makedirs("build", exist_ok=True)
os.makedirs("scripts", exist_ok=True)

# Store GPU architectures and metadata for each kernel
kernel_gpu_archs = []
kernel_metadata = []
layer_kernel_mapping = {}  # Maps layer_id -> list of kernel indices

# Global kernel index across all files
global_kernel_idx = 0

# Statistics tracking
total_configs_processed = 0
total_configs_skipped = 0

# Process each JSON file
for file_idx, json_file in enumerate(json_files):
    layer_id = extract_layer_name_from_filename(json_file)
    layer_name = extract_short_layer_name(json_file)

    print(f"\n{'='*80}")
    print(f"Processing file {file_idx+1}/{len(json_files)}: {os.path.basename(json_file)}")
    print(f"Layer ID: {layer_id}")
    print(f"Layer name: {layer_name}")
    print(f"{'='*80}")

    # Create subdirectories for this layer
    layer_kernels_dir = os.path.join("kernels", layer_id)
    layer_outputs_dir = os.path.join("kernel_outputs", layer_id)
    layer_build_dir = os.path.join("build", layer_id)
    layer_scripts_dir = os.path.join("scripts", layer_id)

    os.makedirs(layer_kernels_dir, exist_ok=True)
    os.makedirs(layer_outputs_dir, exist_ok=True)
    os.makedirs(layer_build_dir, exist_ok=True)
    os.makedirs(layer_scripts_dir, exist_ok=True)

    with open(json_file, 'r') as f:
        all_config = f.readlines()

    print(f"Found {len(all_config)} configurations in this file")

    # Track kernel indices for this layer
    layer_kernel_indices = []

    # Process each line in the current JSON file
    for config_idx, line in enumerate(all_config):
        total_configs_processed += 1

        # Extract GPU architecture and execution time first (before extracting workload params)
        data = json.loads(line)

        # Extract execution time from the record (ALREADY in milliseconds)
        exec_time_ms = data['r'][0][0]  # In milliseconds

        # CRITICAL: Filter out invalid configurations (failed tuning attempts)
        # TVM marks failed configs with execution time = 1e+10 ms (essentially infinity)
        if exec_time_ms >= 1e9:  # Use 1e9 as threshold to catch 1e+10 and similar large values
            print(f"  Config {config_idx+1}/{len(all_config)} -> SKIPPED (invalid: exec_time={exec_time_ms:.2e} ms)")
            total_configs_skipped += 1
            continue

        N, H, W, CO, CI, KH, KW, strides, padding = extract_values_from_json(line)

        print(f"  Config {config_idx+1}/{len(all_config)} -> kernel{global_kernel_idx}")
        print(f"    Conv2D: N={N} H={H} W={W} CO={CO} CI={CI} KH={KH} KW={KW} stride={strides} pad={padding}")

        # Extract GPU architecture from target string
        target_str = data['i'][0][1]  # e.g., "cuda -keys=cuda,gpu -arch=sm_89 ..."
        gpu_arch = "sm_75"  # default fallback
        if "-arch=sm_" in target_str:
            # Extract sm_XX from target string
            arch_start = target_str.find("-arch=sm_") + len("-arch=sm_")
            arch_end = target_str.find(" ", arch_start)
            if arch_end == -1:
                arch_end = len(target_str)
            gpu_arch = "sm_" + target_str[arch_start:arch_end]

        # Store GPU architecture for CMakeLists.txt generation
        kernel_gpu_archs.append(gpu_arch)

        # Track this kernel for the layer
        layer_kernel_indices.append(global_kernel_idx)

        # Store metadata for this kernel
        kernel_metadata.append({
            'kernel_idx': global_kernel_idx,
            'file_idx': file_idx,
            'config_idx': config_idx,
            'layer_id': layer_id,
            'layer_name': layer_name,
            'source_file': os.path.basename(json_file),
            'N': N, 'H': H, 'W': W,
            'CO': CO, 'CI': CI,
            'KH': KH, 'KW': KW,
            'stride_h': strides[0], 'stride_w': strides[1],
            'padding_h': padding[0], 'padding_w': padding[1],
            'gpu_arch': gpu_arch,
            'exec_time_ms': exec_time_ms
        })

        task = auto_scheduler.SearchTask(
            func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
        )
        inp, _ = load_record_from_string(line)
        sch, tvm_args = task.compute_dag.apply_steps_from_state(
                inp.state, task.layout_rewrite_option
            )
        ir_module = tvm.lower(sch, tvm_args)
        primfunc = ir_module["main"]
        from tvm.tir.analysis import verify_gpu_code
        valid = verify_gpu_code(primfunc, {"max_shared_memory_per_block": 48*1024, "max_threads_per_block": 1024})
        print(f"    GPU validation: {valid}")

        func = tvm.build(sch, tvm_args, target)
        str_source = func.imported_modules[0].get_source()

        # cut the string, start from the first extern
        str_source = str_source[str_source.find("extern"):]
        # replace "default_function_kernel0" with "default_function_kernel{global_kernel_idx}"
        str_source = str_source.replace("default_function_kernel", f"default_function_kernel{global_kernel_idx}")

        # dump to file in layer subdirectory
        kernel_header_path = os.path.join(layer_kernels_dir, f"kernel{global_kernel_idx}.cuh")
        with open(kernel_header_path, "w") as f:
            f.write(str_headers)
            f.write(str_source)

        # get parallel dimension tile list from the line
        processor = RecordProcessor(line)
        grid = 1
        block = 1
        for each in processor.json_str['i'][processor.IDX_STATE][1]:
            if each[processor.IDX_NODE_NAME] == "SP" and len(each[processor.IDX_LENGTHS]) == 4:

                dim_len = each[processor.IDX_LOOP_EXTENT]
                tile_list = each[processor.IDX_LENGTHS]

                grid *= dim_len/np.prod(tile_list)
                block *= tile_list[1]

        if grid <= 0 or block <= 0:
            print(f"    Warning: Invalid grid={grid}, block={block}, using defaults")
            grid, block = 1, 256

        print(f"    Grid/Block: grid={int(grid)}, block={int(block)}")

        # Generate individual run script for this kernel in layer subdirectory
        run_script = f"""#!/bin/bash
# Auto-generated run script for kernel {global_kernel_idx}
# Source: {os.path.basename(json_file)} - config {config_idx}
# Layer: {layer_name} ({layer_id})
# Parameters: N={N} H={H} W={W} CO={CO} CI={CI} KH={KH} KW={KW} stride_h={strides[0]} stride_w={strides[1]} pad_h={padding[0]} pad_w={padding[1]}
# GPU Architecture: {gpu_arch}
# Original execution time: {exec_time_ms:.6f} ms

echo "Building and running kernel {global_kernel_idx} ({layer_name})..."
echo "Parameters: N={N} H={H} W={W} CO={CO} CI={CI} KH={KH} KW={KW}"
echo "GPU Architecture: {gpu_arch}"

# Get the absolute path to the project root (2 levels up from scripts/layer_id/)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Build executable for this kernel
mkdir -p build/{layer_id}
cd build/{layer_id}

# Compile individual executable
nvcc -O3 -arch={gpu_arch} -std=c++11 \\
    -I../.. -I../../kernels/{layer_id} \\
    -o kernel{global_kernel_idx} ../../kernels/{layer_id}/kernel{global_kernel_idx}.cu ../../main.cpp \\
    -lnvidia-ml

# Run and capture output
echo "Running kernel {global_kernel_idx}..."
./kernel{global_kernel_idx} {N} {H} {W} {CO} {CI} {KH} {KW} {strides[0]} {padding[0]} | tee ../../kernel_outputs/{layer_id}/output_kernel{global_kernel_idx}.txt

cd "$PROJECT_ROOT"
"""

        run_script_path = os.path.join(layer_scripts_dir, f"run_kernel{global_kernel_idx}.sh")
        with open(run_script_path, "w") as f:
            f.write(run_script)

        # Make run script executable
        os.chmod(run_script_path, 0o755)

        # Generate individual .cu file for this kernel in layer subdirectory
        kernel_cu_path = os.path.join(layer_kernels_dir, f"kernel{global_kernel_idx}.cu")
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line_content in lines:
            new_lines.append(line_content)

            if "// insert headers here" in line_content:
                # insert #include "kernel{global_kernel_idx}.cuh"
                new_lines.append(f"#include \"kernel{global_kernel_idx}.cuh\"\n")

            if "// insert kernel call here" in line_content:
                # insert dim3 size_grid and size_block for this kernel
                # NOTE: TVM kernel signature is (output, input, kernel) - NOT (input, kernel, output)!
                new_lines.append(f"    dim3 size_grid({int(grid)},1,1);\n")
                new_lines.append(f"    dim3 size_block({int(block)},1,1);\n")
                new_lines.append(f"    default_function_kernel{global_kernel_idx} <<< size_grid, size_block >>>(dev_output_ptr, dev_input_ptr, dev_kernel_ptr);\n")

        with open(kernel_cu_path, "w") as f:
            f.writelines(new_lines)

        # Increment global kernel index
        global_kernel_idx += 1

    # Store layer mapping
    layer_kernel_mapping[layer_id] = layer_kernel_indices

    # Generate layer-level run script (runs all configs for this layer)
    layer_run_script = f"""#!/bin/bash
# Run all {len(layer_kernel_indices)} kernels for layer: {layer_name} ({layer_id})

echo "Running all {len(layer_kernel_indices)} kernels for {layer_name}..."
echo "{'='*60}"

"""
    for kid in layer_kernel_indices:
        layer_run_script += f"bash scripts/{layer_id}/run_kernel{kid}.sh\n"

    layer_run_script += f"""
echo ""
echo "All kernels for {layer_name} completed!"
"""

    layer_run_script_path = f"run_layer_{layer_id}.sh"
    with open(layer_run_script_path, "w") as f:
        f.write(layer_run_script)
    os.chmod(layer_run_script_path, 0o755)
    print(f"  Generated layer run script: {layer_run_script_path}")

total_kernels = global_kernel_idx

print(f"\n{'='*80}")
print(f"Kernel generation complete!")
print(f"Total kernels generated: {total_kernels}")
print(f"Total layers: {len(json_files)}")
print(f"{'='*80}\n")

# Write metadata CSV
metadata_csv_path = "kernel_metadata.csv"
print(f"Writing metadata to {metadata_csv_path}...")
with open(metadata_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['kernel_idx', 'file_idx', 'config_idx', 'layer_id', 'layer_name', 'source_file',
                  'N', 'H', 'W', 'CO', 'CI', 'KH', 'KW',
                  'stride_h', 'stride_w', 'padding_h', 'padding_w',
                  'gpu_arch', 'exec_time_ms']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for metadata in kernel_metadata:
        writer.writerow(metadata)

print(f"Metadata written successfully ({len(kernel_metadata)} entries)")

# Generate CMakeLists.txt for building all kernels
print(f"\nGenerating CMakeLists.txt...")
cmake_content = """cmake_minimum_required(VERSION 3.12)

project(tvm-dump LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "-O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

"""

# Add executable for each kernel with layer-specific include path
for idx in range(total_kernels):
    meta = kernel_metadata[idx]
    layer_id = meta['layer_id']
    gpu_arch = kernel_gpu_archs[idx]
    arch_num = gpu_arch.replace("sm_", "")

    cmake_content += f"# Kernel {idx} - {meta['layer_name']}\n"
    cmake_content += f"include_directories(${{CMAKE_SOURCE_DIR}}/kernels/{layer_id})\n"
    cmake_content += f"add_executable(kernel{idx} kernels/{layer_id}/kernel{idx}.cu main.cpp)\n"
    cmake_content += f"set_property(TARGET kernel{idx} PROPERTY CUDA_ARCHITECTURES {arch_num})\n"
    cmake_content += f"target_link_libraries(kernel{idx} nvidia-ml)\n"
    cmake_content += f"set_target_properties(kernel{idx} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/{layer_id})\n\n"

cmake_content += "# Build all kernels target\n"
cmake_content += "add_custom_target(all_kernels)\n"
for idx in range(total_kernels):
    cmake_content += f"add_dependencies(all_kernels kernel{idx})\n"

with open("CMakeLists.txt", "w") as f:
    f.write(cmake_content)

print(f"CMakeLists.txt generated successfully")

# Generate master run script to execute all kernels
print(f"\nGenerating master run scripts...")
master_script = f"""#!/bin/bash
# Master script to run all generated kernels
# Total: {total_kernels} kernels across {len(json_files)} layers
# WARNING: This will run ALL kernels sequentially, which may take a very long time!
# Consider using batch scripts (run_batch_*.sh) or layer scripts (run_layer_*.sh) instead

echo "Running all {total_kernels} generated kernels from {len(json_files)} layers..."
echo "{'='*60}"

"""

for layer_id in sorted(layer_kernel_mapping.keys()):
    master_script += f"\necho \"\"\necho \"Running layer: {layer_id}...\"\nbash run_layer_{layer_id}.sh\n"

master_script += f"""
echo ""
echo "All kernels completed!"
"""

with open("run_all.sh", "w") as f:
    f.write(master_script)

os.chmod("run_all.sh", 0o755)

# Generate batch run scripts (10 batches)
num_batches = 10
batch_size = (total_kernels + num_batches - 1) // num_batches  # Ceiling division

print(f"Generating {num_batches} batch run scripts...")
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, total_kernels)

    if start_idx >= total_kernels:
        break

    batch_script = f"""#!/bin/bash
# Batch script {batch_idx}: runs kernels {start_idx} to {end_idx-1}
# Total: {end_idx - start_idx} kernels

echo "Running batch {batch_idx} (kernels {start_idx}-{end_idx-1})..."
echo "{'='*60}"

"""

    for idx in range(start_idx, end_idx):
        meta = kernel_metadata[idx]
        layer_id = meta['layer_id']
        batch_script += f"\necho \"\"\necho \"Running kernel {idx} ({idx - start_idx + 1}/{end_idx - start_idx} in batch {batch_idx})...\"\nbash scripts/{layer_id}/run_kernel{idx}.sh\n"

    batch_script += f"""
echo ""
echo "Batch {batch_idx} completed!"
"""

    batch_script_path = f"run_batch_{batch_idx}.sh"
    with open(batch_script_path, "w") as f:
        f.write(batch_script)

    os.chmod(batch_script_path, 0o755)
    print(f"  Created {batch_script_path} (kernels {start_idx}-{end_idx-1})")

print(f"\nGeneration complete!")
print(f"{'='*80}")
print(f"Summary:")
print(f"  - Energy measurement: {num_lrounds} lrounds × 100,000 executions per lround = {total_executions:,} total")
print(f"")
print(f"Configuration filtering:")
print(f"  - Total configs processed: {total_configs_processed}")
print(f"  - Valid configs (generated): {total_kernels}")
print(f"  - Invalid configs (skipped): {total_configs_skipped}")
print(f"  - Success rate: {100.0 * total_kernels / total_configs_processed:.1f}%")
print(f"")
print(f"Kernel statistics:")
print(f"  - Total kernels generated: {total_kernels}")
print(f"  - Total layers: {len(json_files)}")
print(f"  - Kernels per layer: ~{total_kernels // len(json_files)}")
print(f"  - GPU architectures: {', '.join(sorted(set(kernel_gpu_archs)))}")
print(f"")
print(f"Directory structure:")
print(f"  kernels/")
for layer_id in sorted(layer_kernel_mapping.keys())[:3]:
    print(f"    {layer_id}/ ({len(layer_kernel_mapping[layer_id])} kernels)")
print(f"    ... ({len(json_files)} layer directories total)")
print(f"")
print(f"  kernel_outputs/")
print(f"    (same structure as kernels/)")
print(f"")
print(f"  build/")
print(f"    (same structure as kernels/)")
print(f"")
print(f"  scripts/")
print(f"    (same structure as kernels/)")
print(f"")
print(f"Layer-level scripts:")
for layer_id in sorted(layer_kernel_mapping.keys())[:3]:
    print(f"  run_layer_{layer_id}.sh")
print(f"  ... ({len(json_files)} layer scripts total)")
print(f"")
print(f"Batch scripts: run_batch_0.sh through run_batch_{num_batches-1}.sh")
print(f"Master script: run_all.sh")
print(f"Metadata: {metadata_csv_path}")
print(f"")
print(f"To build and run:")
print(f"  Single kernel:  bash scripts/<layer_id>/run_kernel<N>.sh")
print(f"  Entire layer:   bash run_layer_<layer_id>.sh")
print(f"  Batch:          bash run_batch_0.sh")
print(f"  All (WARNING):  bash run_all.sh")
print(f"  CMake build:    mkdir -p build && cd build && cmake .. && make -j")
print(f"{'='*80}")
