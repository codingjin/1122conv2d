#!/usr/bin/env python3
"""
Extract energy consumption data from kernel output files and create CSV files.

This script:
1. Creates exp_sketch_dump/energydata/ folder
2. Processes each Conv2D layer folder in kernel_outputs/
3. Extracts energy and std/mean data from output_kernel*.txt files
4. Matches with performance data from performance/*.json files using kernel_metadata.csv
5. Generates CSV files with format: id,perf(GFLOPS),Energy(mj),std/mean(%)
"""

import os
import re
import glob
import csv
from pathlib import Path


def load_kernel_metadata(metadata_csv_path):
    """
    Load kernel metadata from CSV file.

    Args:
        metadata_csv_path: Path to kernel_metadata.csv

    Returns:
        Dict mapping (layer_id, kernel_idx) -> config_idx
    """
    metadata = {}
    try:
        with open(metadata_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel_idx = int(row['kernel_idx'])
                config_idx = int(row['config_idx'])
                layer_id = row['layer_id']
                metadata[(layer_id, kernel_idx)] = config_idx
        return metadata
    except Exception as e:
        print(f"  Error loading kernel metadata: {e}")
        return {}


def extract_energy_from_output(filepath):
    """
    Extract energy per iteration and coefficient of variation from output file.

    Args:
        filepath: Path to output_kernel*.txt file

    Returns:
        Tuple of (energy_mj, std_mean_percent) or (None, None) if extraction fails
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract "Energy per iteration: X.XXX mJ"
        energy_match = re.search(r'Energy per iteration:\s+([\d.]+)\s+mJ', content)

        # Extract "Coefficient of variation (std/mean): X.XXXXX (X.XX%)"
        cv_match = re.search(r'Coefficient of variation \(std/mean\):\s+[\d.]+\s+\(([\d.]+)%\)', content)

        if energy_match and cv_match:
            energy_mj = float(energy_match.group(1))
            std_mean_percent = float(cv_match.group(1))
            return energy_mj, std_mean_percent
        else:
            return None, None

    except Exception as e:
        print(f"  Warning: Failed to extract data from {filepath}: {e}")
        return None, None


def load_performance_data(perf_json_path):
    """
    Load performance data from JSON file.

    Args:
        perf_json_path: Path to performance JSON file

    Returns:
        List of performance values (integers in GFLOPS)
    """
    try:
        with open(perf_json_path, 'r') as f:
            lines = f.readlines()
            # Each line is a single integer GFLOPS value
            perf_data = [int(line.strip()) for line in lines if line.strip()]
            return perf_data
    except Exception as e:
        print(f"  Warning: Failed to load performance data from {perf_json_path}: {e}")
        return []


def get_sorted_kernel_files(kernel_output_dir):
    """
    Get sorted list of output_kernel*.txt files.

    Args:
        kernel_output_dir: Directory containing output files

    Returns:
        List of (kernel_number, filepath) tuples, sorted by kernel_number
    """
    pattern = os.path.join(kernel_output_dir, "output_kernel*.txt")
    files = glob.glob(pattern)

    # Extract kernel numbers and sort
    kernel_files = []
    for filepath in files:
        basename = os.path.basename(filepath)
        match = re.search(r'output_kernel(\d+)\.txt', basename)
        if match:
            kernel_num = int(match.group(1))
            kernel_files.append((kernel_num, filepath))

    # Sort by kernel number
    kernel_files.sort(key=lambda x: x[0])
    return kernel_files


def find_performance_file(layer_id, performance_dir):
    """
    Find the corresponding performance JSON file for a layer.

    Args:
        layer_id: Layer folder name (e.g., "cuda_resnet_testCase_0")
        performance_dir: Directory containing performance JSON files

    Returns:
        Path to performance JSON file or None if not found
    """
    # Performance files are named like:
    # cuda_resnet_testCase_0_conv2d_N_1_H_224_W_224_CO_64_CI_3_KH_7_KW_7_strides_(2, 2)_padding_(3, 3).json
    pattern = os.path.join(performance_dir, f"{layer_id}_*.json")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]
    else:
        return None


def process_layer(layer_id, kernel_output_dir, performance_dir, output_csv_path, metadata):
    """
    Process a single Conv2D layer and generate CSV file.

    Args:
        layer_id: Layer folder name
        kernel_output_dir: Directory containing kernel output files
        performance_dir: Directory containing performance JSON files
        output_csv_path: Output CSV file path
        metadata: Dict mapping (layer_id, kernel_idx) -> config_idx
    """
    print(f"\nProcessing layer: {layer_id}")

    # Find performance file
    perf_file = find_performance_file(layer_id, performance_dir)
    if not perf_file:
        print(f"  Error: Performance file not found for {layer_id}")
        return

    print(f"  Performance file: {os.path.basename(perf_file)}")

    # Load performance data
    perf_data = load_performance_data(perf_file)
    if not perf_data:
        print(f"  Error: Failed to load performance data")
        return

    print(f"  Performance data loaded: {len(perf_data)} entries")

    # Get sorted kernel output files
    kernel_files = get_sorted_kernel_files(kernel_output_dir)
    print(f"  Kernel output files found: {len(kernel_files)}")

    if not kernel_files:
        print(f"  Error: No kernel output files found")
        return

    # Extract data and create CSV
    csv_data = []
    valid_count = 0
    invalid_count = 0

    for kernel_idx, filepath in kernel_files:
        # Extract energy data
        energy_mj, std_mean_percent = extract_energy_from_output(filepath)

        if energy_mj is None or std_mean_percent is None:
            print(f"  Skipping kernel {kernel_idx}: Failed to extract energy data")
            invalid_count += 1
            continue

        # Get config_idx from metadata
        key = (layer_id, kernel_idx)
        if key not in metadata:
            print(f"  Skipping kernel {kernel_idx}: Not found in metadata")
            invalid_count += 1
            continue

        config_idx = metadata[key]

        # Get corresponding performance data
        if config_idx < len(perf_data):
            perf_gflops = perf_data[config_idx]
        else:
            print(f"  Skipping kernel {kernel_idx} (config_idx={config_idx}): No corresponding performance data")
            invalid_count += 1
            continue

        # Add to CSV data (use config_idx as the id for ordering)
        csv_data.append({
            'id': config_idx,
            'perf(GFLOPS)': perf_gflops,
            'Energy(mj)': energy_mj,
            'std/mean(%)': std_mean_percent
        })
        valid_count += 1

    # Sort by id (config_idx) to maintain order
    csv_data.sort(key=lambda x: x['id'])

    # Write CSV file
    if csv_data:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'perf(GFLOPS)', 'Energy(mj)', 'std/mean(%)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

        print(f"  CSV created: {output_csv_path}")
        print(f"  Valid records: {valid_count}, Invalid records skipped: {invalid_count}")
    else:
        print(f"  Error: No valid data to write")


def main():
    """Main function to process all layers."""

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_sketch_dump_dir = os.path.join(script_dir, "exp_sketch_dump")
    kernel_outputs_dir = os.path.join(exp_sketch_dump_dir, "kernel_outputs")
    performance_dir = os.path.join(script_dir, "performance")
    energydata_dir = os.path.join(exp_sketch_dump_dir, "energydata")
    metadata_csv_path = os.path.join(exp_sketch_dump_dir, "kernel_metadata.csv")

    print("="*80)
    print("Energy Data Extraction Script")
    print("="*80)
    print(f"Kernel outputs directory: {kernel_outputs_dir}")
    print(f"Performance directory: {performance_dir}")
    print(f"Metadata file: {metadata_csv_path}")
    print(f"Output directory: {energydata_dir}")

    # Create energydata directory if it doesn't exist
    os.makedirs(energydata_dir, exist_ok=True)
    print(f"\nCreated/verified output directory: {energydata_dir}")

    # Check if directories exist
    if not os.path.exists(kernel_outputs_dir):
        print(f"\nError: Kernel outputs directory not found: {kernel_outputs_dir}")
        return

    if not os.path.exists(performance_dir):
        print(f"\nError: Performance directory not found: {performance_dir}")
        return

    if not os.path.exists(metadata_csv_path):
        print(f"\nError: Metadata file not found: {metadata_csv_path}")
        return

    # Load kernel metadata
    print("\nLoading kernel metadata...")
    metadata = load_kernel_metadata(metadata_csv_path)
    print(f"Loaded metadata for {len(metadata)} kernels")

    if not metadata:
        print("Error: Failed to load kernel metadata")
        return

    # Get all layer folders
    layer_folders = []
    for item in os.listdir(kernel_outputs_dir):
        item_path = os.path.join(kernel_outputs_dir, item)
        if os.path.isdir(item_path):
            layer_folders.append(item)

    layer_folders.sort()
    print(f"\nFound {len(layer_folders)} layer folders to process")

    # Process each layer
    successful_csvs = 0
    for i, layer_id in enumerate(layer_folders, 1):
        print(f"\n{'='*80}")
        print(f"Processing layer {i}/{len(layer_folders)}: {layer_id}")
        print(f"{'='*80}")

        kernel_output_dir = os.path.join(kernel_outputs_dir, layer_id)
        output_csv_path = os.path.join(energydata_dir, f"{layer_id}.csv")

        process_layer(layer_id, kernel_output_dir, performance_dir, output_csv_path, metadata)

        # Check if CSV was created
        if os.path.exists(output_csv_path):
            successful_csvs += 1

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)
    print(f"\nCSV files created in: {energydata_dir}")

    # List created CSV files
    csv_files = glob.glob(os.path.join(energydata_dir, "*.csv"))
    print(f"Total CSV files created: {len(csv_files)} out of {len(layer_folders)} layers")


if __name__ == "__main__":
    main()
