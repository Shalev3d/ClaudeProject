#!/usr/bin/env python3
"""
Test FPGA cycle counting with small matrices that fit FPGA constraints
"""

import torch
import numpy as np
import time
from k5_fpga_accelerator import K5FPGAAccelerator, fpga_matmul


def test_small_matrix_fpga_cycles():
    """Test FPGA operations with small matrices to see actual FPGA cycle usage"""
    print("🔍 Testing FPGA Cycles with Small Matrices")
    print("=" * 50)
    
    # Initialize FPGA accelerator
    fpga_accelerator = K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier")
    
    # Test different matrix sizes within FPGA limits (8x8 max)
    test_sizes = [(2, 2), (4, 4), (6, 6), (8, 8)]
    
    for rows, cols in test_sizes:
        print(f"\n🧮 Testing {rows}x{cols} Matrix Multiplication")
        print("-" * 40)
        
        # Create small test matrices
        A = torch.randn(rows, cols, dtype=torch.float32)
        B = torch.randn(cols, rows, dtype=torch.float32)  # Transpose for valid multiplication
        
        print(f"Matrix A shape: {A.shape}")
        print(f"Matrix B shape: {B.shape}")
        print(f"Expected result shape: ({rows}, {rows})")
        
        # Reset FPGA statistics
        fpga_accelerator.reset_stats()
        
        # Measure CPU computation
        start_cpu = time.perf_counter()
        cpu_result = torch.matmul(A, B)
        cpu_time = time.perf_counter() - start_cpu
        
        # Measure FPGA computation
        start_fpga = time.perf_counter()
        fpga_result = fpga_matmul(A, B)
        fpga_time = time.perf_counter() - start_fpga
        
        # Get FPGA statistics
        stats = fpga_accelerator.get_performance_stats()
        
        # Verify correctness
        error = torch.mean(torch.abs(cpu_result - fpga_result)).item()
        
        print(f"✅ Results:")
        print(f"   • CPU time: {cpu_time*1000:.3f} ms")
        print(f"   • FPGA time: {fpga_time*1000:.3f} ms")
        print(f"   • Speedup: {cpu_time/fpga_time:.2f}x {'🚀' if cpu_time > fpga_time else '📉'}")
        print(f"   • FPGA operations: {stats['fpga_calls']}")
        print(f"   • CPU fallbacks: {stats['cpu_fallback_calls']}")
        print(f"   • FPGA usage ratio: {stats['fpga_usage_ratio']:.1%}")
        print(f"   • Average error: {error:.6f}")
        
        if stats['fpga_calls'] > 0:
            print(f"   • Average FPGA op time: {stats['fpga_time_average']*1000:.3f} ms")
            
            # Estimate FPGA cycles
            # Assume 100MHz FPGA clock
            fpga_clock_freq = 100e6  # 100 MHz
            estimated_fpga_cycles = stats['fpga_time_total'] * fpga_clock_freq
            
            # CPU cycles (rough estimate)
            cpu_freq = 4e9  # 4 GHz (typical)
            estimated_cpu_cycles = cpu_time * cpu_freq
            
            print(f"📊 Cycle Estimates:")
            print(f"   • CPU cycles: {estimated_cpu_cycles:,.0f}")
            print(f"   • FPGA cycles: {estimated_fpga_cycles:,.0f}")
            print(f"   • FPGA efficiency: {estimated_cpu_cycles/estimated_fpga_cycles:.2f}x")
        else:
            print("   ⚠️  No FPGA operations performed")
        
        print(f"   • Correctness check: {'✅ PASS' if error < 0.01 else '❌ FAIL'}")


def test_fpga_matrix_size_limits():
    """Test what matrix sizes can actually use the FPGA"""
    print(f"\n🔍 Testing FPGA Matrix Size Limits")
    print("=" * 50)
    
    fpga_accelerator = K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier")
    
    # Test various sizes to find the limit
    test_cases = [
        (2, 2, 2),   # Very small
        (4, 4, 4),   # Small  
        (6, 6, 6),   # Medium small
        (8, 8, 8),   # At limit
        (10, 10, 10), # Over limit
        (16, 16, 16), # Much over limit
        (32, 32, 32), # Transformer typical
    ]
    
    for m, k, n in test_cases:
        A = torch.randn(m, k, dtype=torch.float32)
        B = torch.randn(k, n, dtype=torch.float32)
        
        suitable = fpga_accelerator.is_fpga_suitable(A, B)
        
        print(f"Matrix ({m}×{k}) @ ({k}×{n}): {'✅ FPGA suitable' if suitable else '❌ CPU fallback'}")
        
        if suitable:
            fpga_accelerator.reset_stats()
            result = fpga_matmul(A, B)
            stats = fpga_accelerator.get_performance_stats()
            print(f"   • FPGA operations: {stats['fpga_calls']}")
            print(f"   • Actual FPGA usage: {stats['fpga_usage_ratio']:.1%}")


if __name__ == "__main__":
    try:
        test_small_matrix_fpga_cycles()
        test_fpga_matrix_size_limits()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise