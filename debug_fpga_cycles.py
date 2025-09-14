#!/usr/bin/env python3
"""
Debug FPGA cycle execution to understand why operations aren't being performed
"""

import torch
import numpy as np
import os
from k5_fpga_accelerator import K5FPGAAccelerator


def debug_fpga_execution():
    """Debug step-by-step FPGA execution"""
    print("🔍 Debugging FPGA Execution Step-by-Step")
    print("=" * 50)
    
    # Create FPGA accelerator
    fpga_accelerator = K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier")
    
    # Simple 2x2 matrix test
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    
    print(f"Testing with matrices:")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"Expected result (A @ B) = {torch.matmul(A, B)}")
    
    # Check if suitable for FPGA
    suitable = fpga_accelerator.is_fpga_suitable(A, B)
    print(f"\nFPGA suitable: {suitable}")
    
    if not suitable:
        print("❌ Matrix not suitable for FPGA - stopping debug")
        return
    
    # Step through the FPGA process
    print(f"\n📝 Step 1: Convert tensors to int16")
    scale = 100.0
    A_int16 = fpga_accelerator.tensor_to_int16(A, scale)
    B_int16 = fpga_accelerator.tensor_to_int16(B, scale)
    
    print(f"A_int16 = {A_int16}")
    print(f"B_int16 = {B_int16}")
    
    print(f"\n📁 Step 2: Create K5 test files")
    try:
        temp_dir = fpga_accelerator.create_k5_test_files(A_int16, B_int16)
        print(f"Created files in: {temp_dir}")
        
        # List files
        files = os.listdir(temp_dir)
        print(f"Files created: {files}")
        
        # Show file contents (first few lines)
        for filename in files:
            filepath = os.path.join(temp_dir, filename)
            print(f"\n📄 Contents of {filename}:")
            with open(filepath, 'r') as f:
                lines = f.readlines()[:3]  # First 3 lines
                for i, line in enumerate(lines):
                    print(f"   {i+1}: {line.strip()}")
                if len(lines) >= 3:
                    print(f"   ... ({len(lines)} total lines)")
    
    except Exception as e:
        print(f"❌ Failed to create K5 files: {e}")
        return
    
    print(f"\n🚀 Step 3: Run K5 matrix multiplication")
    try:
        result_int16 = fpga_accelerator.run_k5_matrix_multiply(temp_dir)
        print(f"FPGA result (int16): {result_int16}")
        
        # Convert back to float
        result_float = fpga_accelerator.int16_to_tensor(result_int16, scale * scale, 'cpu', torch.float32)
        print(f"FPGA result (float): {result_float}")
        
        # Compare with CPU
        cpu_result = torch.matmul(A, B)
        error = torch.mean(torch.abs(cpu_result - result_float)).item()
        print(f"CPU result: {cpu_result}")
        print(f"Average error: {error:.6f}")
        
        if error < 0.1:
            print("✅ FPGA computation successful!")
        else:
            print("⚠️  FPGA computation has significant error")
            
    except Exception as e:
        print(f"❌ K5 matrix multiplication failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n📊 Step 4: Check FPGA statistics")
    stats = fpga_accelerator.get_performance_stats()
    print(f"FPGA calls: {stats['fpga_calls']}")
    print(f"CPU fallbacks: {stats['cpu_fallback_calls']}")
    print(f"Total FPGA time: {stats['fpga_time_total']:.4f} seconds")
    
    if stats['fpga_calls'] > 0:
        # Estimate cycles
        fpga_freq = 100e6  # 100 MHz
        fpga_cycles = stats['fpga_time_total'] * fpga_freq
        print(f"Estimated FPGA cycles: {fpga_cycles:,.0f}")
        
        # Break down cycles
        print(f"\n🔍 Cycle Breakdown Estimate:")
        print(f"   • Total FPGA time: {stats['fpga_time_total']*1000:.1f} ms")
        print(f"   • At 100 MHz: {fpga_cycles:,.0f} cycles")
        print(f"   • Pure computation estimate: ~1000 cycles")
        print(f"   • Communication overhead: ~{fpga_cycles-1000:,.0f} cycles")
    else:
        print("❌ No FPGA operations recorded")


if __name__ == "__main__":
    debug_fpga_execution()