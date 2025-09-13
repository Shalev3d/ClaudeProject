#!/usr/bin/env python3
"""
Test script for FPGA Matrix Multiplier Accelerator
Tests the complete pipeline from Python to FPGA and back
"""

import numpy as np
import torch
import torch.nn as nn
import logging
import time
from fpga_accelerator import FPGAAccelerator, FPGALinear
from model import build_transformer
from config import get_config

def test_fpga_connection():
    """Test basic FPGA connection and communication"""
    print("=== Testing FPGA Connection ===")
    
    fpga = FPGAAccelerator(port='/dev/ttyUSB0')  # Adjust port as needed
    
    if not fpga.connect():
        print("❌ FPGA connection failed")
        return False
    
    print("✅ FPGA connected successfully")
    fpga.disconnect()
    return True

def test_matrix_multiplication():
    """Test matrix multiplication accuracy"""
    print("\n=== Testing Matrix Multiplication ===")
    
    fpga = FPGAAccelerator(port='/dev/ttyUSB0')
    
    if not fpga.connect():
        print("❌ FPGA connection failed")
        return False
    
    # Test different matrix sizes
    test_sizes = [(4, 6, 8), (8, 8, 8), (16, 12, 10)]
    
    for M, K, N in test_sizes:
        print(f"\nTesting {M}x{K} @ {K}x{N} matrix multiplication...")
        
        # Generate random test matrices
        A = np.random.randn(M, K).astype(np.float32) * 0.5  # Scale for fixed-point
        B = np.random.randn(K, N).astype(np.float32) * 0.5
        
        # CPU reference
        start_time = time.time()
        C_cpu = A @ B
        cpu_time = time.time() - start_time
        
        # FPGA computation
        start_time = time.time()
        try:
            C_fpga = fpga.matrix_multiply(A, B)
            fpga_time = time.time() - start_time
            
            # Compare results
            error = np.mean(np.abs(C_cpu - C_fpga))
            max_error = np.max(np.abs(C_cpu - C_fpga))
            
            print(f"  Mean absolute error: {error:.6f}")
            print(f"  Max absolute error: {max_error:.6f}")
            print(f"  CPU time: {cpu_time*1000:.2f} ms")
            print(f"  FPGA time: {fpga_time*1000:.2f} ms")
            
            if error < 0.1:  # Tolerance for fixed-point precision
                print("  ✅ Test passed")
            else:
                print("  ❌ Test failed - error too large")
                return False
                
        except Exception as e:
            print(f"  ❌ FPGA computation failed: {e}")
            return False
    
    fpga.disconnect()
    return True

def test_fpga_linear_layer():
    """Test FPGA-accelerated linear layer"""
    print("\n=== Testing FPGA Linear Layer ===")
    
    fpga = FPGAAccelerator(port='/dev/ttyUSB0')
    
    if not fpga.connect():
        print("❌ FPGA connection failed")
        return False
    
    # Test parameters
    batch_size = 2
    seq_len = 16
    in_features = 32
    out_features = 64
    
    # Create test input
    input_tensor = torch.randn(batch_size, seq_len, in_features)
    
    # Create CPU and FPGA linear layers with same weights
    cpu_layer = nn.Linear(in_features, out_features)
    fpga_layer = FPGALinear(in_features, out_features, fpga_accelerator=fpga)
    
    # Copy weights
    fpga_layer.weight.data = cpu_layer.weight.data.clone()
    if cpu_layer.bias is not None:
        fpga_layer.bias.data = cpu_layer.bias.data.clone()
    
    # Forward pass
    print(f"Input shape: {input_tensor.shape}")
    
    start_time = time.time()
    cpu_output = cpu_layer(input_tensor)
    cpu_time = time.time() - start_time
    
    start_time = time.time()
    fpga_output = fpga_layer(input_tensor)
    fpga_time = time.time() - start_time
    
    # Compare results
    error = torch.mean(torch.abs(cpu_output - fpga_output)).item()
    max_error = torch.max(torch.abs(cpu_output - fpga_output)).item()
    
    print(f"Output shape: {cpu_output.shape}")
    print(f"Mean absolute error: {error:.6f}")
    print(f"Max absolute error: {max_error:.6f}")
    print(f"CPU time: {cpu_time*1000:.2f} ms")
    print(f"FPGA time: {fpga_time*1000:.2f} ms")
    
    fpga.disconnect()
    
    if error < 0.1:
        print("✅ FPGA Linear layer test passed")
        return True
    else:
        print("❌ FPGA Linear layer test failed")
        return False

def test_transformer_with_fpga():
    """Test transformer model with FPGA acceleration"""
    print("\n=== Testing Transformer with FPGA ===")
    
    fpga = FPGAAccelerator(port='/dev/ttyUSB0')
    
    if not fpga.connect():
        print("❌ FPGA connection failed")
        return False
    
    # Small transformer for testing
    config = {
        'seq_len': 32,
        'd_model': 64,
        'layers': 1,
        'heads': 2
    }
    
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    # Build models
    cpu_model = build_transformer(
        src_vocab_size, tgt_vocab_size, 
        config['seq_len'], config['seq_len'],
        d_model=config['d_model'], 
        N=config['layers'], 
        h=config['heads']
    )
    
    fpga_model = build_transformer(
        src_vocab_size, tgt_vocab_size,
        config['seq_len'], config['seq_len'], 
        d_model=config['d_model'],
        N=config['layers'],
        h=config['heads'],
        fpga_accelerator=fpga
    )
    
    # Copy weights from CPU model to FPGA model
    fpga_model.load_state_dict(cpu_model.state_dict())
    
    # Test input
    batch_size = 1
    src_seq = torch.randint(0, src_vocab_size, (batch_size, config['seq_len']))
    src_mask = torch.ones(batch_size, 1, 1, config['seq_len'])
    
    print(f"Testing encoder with input shape: {src_seq.shape}")
    
    # Encode with both models
    cpu_model.eval()
    fpga_model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        cpu_encoded = cpu_model.encode(src_seq, src_mask)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        fpga_encoded = fpga_model.encode(src_seq, src_mask)
        fpga_time = time.time() - start_time
    
    # Compare results
    error = torch.mean(torch.abs(cpu_encoded - fpga_encoded)).item()
    max_error = torch.max(torch.abs(cpu_encoded - fpga_encoded)).item()
    
    print(f"Encoded output shape: {cpu_encoded.shape}")
    print(f"Mean absolute error: {error:.6f}")
    print(f"Max absolute error: {max_error:.6f}")
    print(f"CPU encoding time: {cpu_time*1000:.2f} ms")
    print(f"FPGA encoding time: {fpga_time*1000:.2f} ms")
    
    fpga.disconnect()
    
    if error < 0.1:
        print("✅ Transformer FPGA test passed")
        return True
    else:
        print("❌ Transformer FPGA test failed")
        return False

def benchmark_performance():
    """Benchmark FPGA vs CPU performance"""
    print("\n=== Performance Benchmark ===")
    
    fpga = FPGAAccelerator(port='/dev/ttyUSB0')
    
    if not fpga.connect():
        print("❌ FPGA connection failed")
        return False
    
    # Benchmark different sizes
    sizes = [(32, 32, 32), (64, 64, 64), (128, 64, 32)]
    
    print(f"{'Size':<15} {'CPU (ms)':<10} {'FPGA (ms)':<10} {'Speedup':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32) * 0.5
        B = np.random.randn(K, N).astype(np.float32) * 0.5
        
        # Warm up
        _ = A @ B
        _ = fpga.matrix_multiply(A, B)
        
        # CPU benchmark
        cpu_times = []
        for _ in range(5):
            start = time.time()
            C_cpu = A @ B
            cpu_times.append(time.time() - start)
        cpu_time_avg = np.mean(cpu_times) * 1000
        
        # FPGA benchmark
        fpga_times = []
        for _ in range(5):
            start = time.time()
            C_fpga = fpga.matrix_multiply(A, B)
            fpga_times.append(time.time() - start)
        fpga_time_avg = np.mean(fpga_times) * 1000
        
        # Calculate metrics
        speedup = cpu_time_avg / fpga_time_avg if fpga_time_avg > 0 else 0
        error = np.mean(np.abs(C_cpu - C_fpga))
        
        size_str = f"{M}x{K}x{N}"
        print(f"{size_str:<15} {cpu_time_avg:<10.2f} {fpga_time_avg:<10.2f} {speedup:<10.2f}x {error:<10.6f}")
    
    fpga.disconnect()
    return True

def main():
    """Run all tests"""
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 FPGA Matrix Multiplier Test Suite")
    print("=====================================")
    
    tests = [
        ("FPGA Connection", test_fpga_connection),
        ("Matrix Multiplication", test_matrix_multiplication),
        ("FPGA Linear Layer", test_fpga_linear_layer),
        ("Transformer with FPGA", test_transformer_with_fpga),
        ("Performance Benchmark", benchmark_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! FPGA accelerator is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check FPGA connection and implementation.")

if __name__ == "__main__":
    main()