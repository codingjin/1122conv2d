
            #include <cassert>
            #include <stdlib.h>
            #include <cuda.h>
            #include <nvml.h>
            #include <cmath>
            #include <algorithm>
            #include "common.h"
            // insert headers here

void conv_kernel_wrapper(int N_B, int N_C, int N_H, int N_W, int N_F, int N_R, int N_S, int PaddingH, int PaddingW,
                        int StrideH, int StrideW, int N_X, int N_Y, const float *Input,
                        const float *Kernel, float *Output, int itr) {
                        float *dev_Input_all;
                        float *dev_Kernel_all;
                        float *dev_Output_all;

                        // Calculate sizes for single iteration
                        size_t input_size_per_iter = sizeof(float) * N_B * N_C * N_H * N_W;
                        size_t kernel_size_per_iter = sizeof(float) * N_F * N_C * N_R * N_S;
                        size_t output_size_per_iter = sizeof(float) * N_B * N_F * N_Y * N_X;

                        // Allocate GPU memory for ALL iterations upfront
                        printf("Allocating GPU memory for %d iterations...\n", itr);
                        CHECK(cudaMalloc(&dev_Input_all, input_size_per_iter * itr));
                        CHECK(cudaMalloc(&dev_Kernel_all, kernel_size_per_iter * itr));
                        CHECK(cudaMalloc(&dev_Output_all, output_size_per_iter * itr));

                        // Pre-load ALL iteration data to GPU (OUTSIDE energy measurement)
                        printf("Copying all iteration data to GPU...\n");
                        CHECK(cudaMemcpy(dev_Input_all, Input, input_size_per_iter * itr, cudaMemcpyHostToDevice));
                        CHECK(cudaMemcpy(dev_Kernel_all, Kernel, kernel_size_per_iter * itr, cudaMemcpyHostToDevice));
                        CHECK(cudaMemset(dev_Output_all, 0, output_size_per_iter * itr));
                        CHECK(cudaDeviceSynchronize());

                        printf("GPU memory setup complete. Starting energy measurement...\n");
                        fflush(stdout);

                        // Initialize NVML for energy measurement
                        nvmlReturn_t nvml_result;
                        nvmlDevice_t nvml_device;

                        nvml_result = nvmlInit();
                        if (nvml_result != NVML_SUCCESS) {
                            fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(nvml_result));
                        }

                        nvml_result = nvmlDeviceGetHandleByIndex(0, &nvml_device);
                        if (nvml_result != NVML_SUCCESS) {
                            fprintf(stderr, "Failed to get NVML device handle: %s\n", nvmlErrorString(nvml_result));
                        }

                        const int num_lrounds = 1000;  // Will be replaced by Python script (10 or 100)
                        const int num_rounds = 100;    // Fixed for both modes

                        printf("Running %d large rounds, each with %d rounds × %d iterations...\n", num_lrounds, num_rounds, itr);
                        printf("Total executions: %lld kernels\n", (long long)num_lrounds * num_rounds * itr);
                        fflush(stdout);

                        // Array to store energy per large round
                        unsigned long long energy_lrounds[num_lrounds];

                        // Run large rounds (outer loop for statistics)
                        for (int lround = 0; lround < num_lrounds; lround++) {
                            unsigned long long energy_start, energy_end;

                            // Get energy BEFORE this large round
                            nvml_result = nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_start);
                            if (nvml_result != NVML_SUCCESS) {
                                fprintf(stderr, "Failed to get start energy: %s\n", nvmlErrorString(nvml_result));
                                energy_start = 0;
                            }

                            // Run rounds within this large round
                            for (int round = 0; round < num_rounds; round++) {
                                // Run kernel for itr iterations - ONLY kernel launches, NO memory operations
                                for (int i = 0; i < itr; i++) {
                                    // Calculate GPU memory offsets for this iteration (no host pointers needed)
                                    float *dev_input_ptr = dev_Input_all + i * N_B * N_C * N_H * N_W;
                                    float *dev_kernel_ptr = dev_Kernel_all + i * N_F * N_C * N_R * N_S;
                                    float *dev_output_ptr = dev_Output_all + i * N_B * N_F * N_Y * N_X;

                    // insert kernel call here

                                }
                            }

                            // Synchronize to ensure all kernels in this large round completed
                            CHECK(cudaDeviceSynchronize());

                            // Get energy AFTER this large round
                            nvml_result = nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_end);
                            if (nvml_result != NVML_SUCCESS) {
                                fprintf(stderr, "Failed to get end energy: %s\n", nvmlErrorString(nvml_result));
                                energy_end = energy_start;
                            }

                            // Store energy for this large round
                            energy_lrounds[lround] = energy_end - energy_start;
                        }

                        // Calculate statistics across large rounds
                        long long kernels_per_lround = (long long)num_rounds * itr;
                        long long total_kernel_executions = (long long)num_lrounds * num_rounds * itr;

                        // Calculate mean energy per large round
                        double sum_energy = 0.0;
                        for (int lr = 0; lr < num_lrounds; lr++) {
                            sum_energy += (double)energy_lrounds[lr];
                        }
                        double mean_energy_lround = sum_energy / num_lrounds;

                        // Calculate standard deviation
                        double variance = 0.0;
                        for (int lr = 0; lr < num_lrounds; lr++) {
                            double diff = (double)energy_lrounds[lr] - mean_energy_lround;
                            variance += diff * diff;
                        }
                        variance /= num_lrounds;
                        double std_dev_lround = sqrt(variance);

                        // Calculate coefficient of variation (CV = std/mean)
                        double cv_lround = std_dev_lround / mean_energy_lround;

                        // Calculate total energy
                        unsigned long long total_energy = 0;
                        for (int lr = 0; lr < num_lrounds; lr++) {
                            total_energy += energy_lrounds[lr];
                        }

                        // Print statistics
                        printf("\n=== Energy Measurement Results (%d lrounds × %d rounds × %d iterations) ===\n", num_lrounds, num_rounds, itr);
                        printf("\n[Total Energy Consumed]\n");
                        printf("  Total executions: %lld kernels\n", total_kernel_executions);
                        printf("  Total energy: %llu mJ (%.6f J)\n", total_energy, (double)total_energy / 1000.0);

                        printf("\n[Large Round Statistics] (%lld kernel executions per large round)\n", kernels_per_lround);
                        printf("  Average energy per large round: %.6f mJ (%.9f J)\n", mean_energy_lround, mean_energy_lround / 1000.0);
                        printf("  Standard deviation: %.6f mJ (%.9f J)\n", std_dev_lround, std_dev_lround / 1000.0);
                        printf("  Coefficient of variation (std/mean): %.6f (%.2f%%)\n", cv_lround, cv_lround * 100.0);

                        printf("\n[Average Energy per Round] (%d kernel executions)\n", itr);
                        printf("  Energy per round: %.6f mJ (%.9f J)\n", mean_energy_lround / num_rounds, mean_energy_lround / num_rounds / 1000.0);

                        printf("\n[Average Energy per Iteration] (single kernel execution)\n");
                        printf("  Energy per iteration: %.9f mJ (%.12f J)\n", mean_energy_lround / kernels_per_lround, mean_energy_lround / kernels_per_lround / 1000.0);

                        printf("\nNote: NVML API returns integer mJ. Averaged values show computed precision.\n");
                        printf("==================================================================\n");
                        fflush(stdout);

                        // Cleanup NVML
                        nvmlShutdown();

                        // Optionally copy results back to host (outside energy measurement)
                        // Uncomment if you need to verify outputs:
                        // CHECK(cudaMemcpy(Output, dev_Output_all, output_size_per_iter * itr, cudaMemcpyDeviceToHost));

                        // Free GPU memory
                        CHECK(cudaFree(dev_Input_all));
                        CHECK(cudaFree(dev_Kernel_all));
                        CHECK(cudaFree(dev_Output_all));

                    }
            