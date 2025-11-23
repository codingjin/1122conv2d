#!/bin/bash
# Clean script to remove all generated files from measure_single_sketch.py and measure_all_tuning_results.py
# This prepares the project for clean git commits

echo "Cleaning generated files..."

# Remove generated kernel files (hierarchical structure)
if [ -d "kernels" ]; then
    echo "  Removing kernels/ directory (with all layer subdirectories)..."
    rm -rf kernels/
fi

# Remove build directory (hierarchical structure)
if [ -d "build" ]; then
    echo "  Removing build/ directory (with all layer subdirectories)..."
    rm -rf build/
fi

# Remove kernel output directory (hierarchical structure)
if [ -d "kernel_outputs" ]; then
    echo "  Removing kernel_outputs/ directory (with all layer subdirectories)..."
    rm -rf kernel_outputs/
fi

# Remove scripts directory (hierarchical structure)
if [ -d "scripts" ]; then
    echo "  Removing scripts/ directory (with all layer subdirectories)..."
    rm -rf scripts/
fi

# Remove generated run scripts (individual kernels - legacy from flat structure)
echo "  Removing generated run scripts..."
rm -f run_kernel*.sh

# Remove layer-level run scripts (new hierarchical structure)
echo "  Removing layer-level run scripts..."
rm -f run_layer_*.sh

# Remove batch run scripts
echo "  Removing batch run scripts..."
rm -f run_batch_*.sh

# Remove master run script
rm -f run_all.sh

# Remove metadata file
if [ -f "kernel_metadata.csv" ]; then
    echo "  Removing kernel_metadata.csv..."
    rm -f kernel_metadata.csv
fi

# Remove auto-generated CMakeLists.txt
if [ -f "CMakeLists.txt" ]; then
    echo "  Removing auto-generated CMakeLists.txt..."
    rm -f CMakeLists.txt
fi

# Remove any accidentally generated kernel files in root directory
#rm -f kernel*.cu kernel*.cuh
#rm -f Resnet10_all.cu

echo ""
echo "Cleanup complete!"
echo ""
echo "Kept files (templates and scripts):"
echo "  - demo.cu (kernel wrapper template)"
echo "  - main.cpp (entry point)"
echo "  - common.h (shared utilities)"
echo "  - sketch.json (single sketch input)"
echo "  - measure_single_sketch.py (single sketch processor)"
echo "  - measure_all_tuning_results.py (full tuning results processor)"
echo "  - test_pipeline.py (testing script)"
echo "  - test_hierarchical.py (hierarchical structure test)"
echo "  - CLAUDE.md (documentation)"
echo "  - clean.sh (this cleanup script)"
echo ""
echo "To regenerate kernels:"
echo "  Test mode:  python measure_all_tuning_results.py --test"
echo "  Full mode:  python measure_all_tuning_results.py"
