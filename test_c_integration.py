#!/usr/bin/env python3
"""
Test the C program integration with Python
Verify that the enhanced C program can be called from Python transformer training
"""

import torch
import numpy as np
import os
import tempfile
from k5_fpga_accelerator import K5FPGAAccelerator


def test_c_program_integration():
    """Test calling the C program directly from Python"""
    print("🧪 Testing C Program Integration")
    print("=" * 50)
    
    # Check if C program exists
    c_program_path = "./de10_lite_matrix_multiplier"
    c_program_exists = os.path.isfile(c_program_path)
    
    print(f"C program path: {c_program_path}")
    print(f"C program exists: {c_program_exists}")
    
    if not c_program_exists:
        print("⚠️  C program not found - this will test simulation mode")
        print("   To test with real hardware:")
        print("   1. Compile on K5 system: gcc -o de10_lite_matrix_multiplier de10_lite_matrix_multiplier.c -lk5")
        print("   2. Copy executable to this directory")
    
    # Test small matrix multiplication
    print(f"\n📊 Testing 2x2 Matrix Multiplication")
    
    # Create test matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    
    print(f"Matrix A: {A}")
    print(f"Matrix B: {B}")
    
    # CPU reference result
    cpu_result = torch.matmul(A, B)
    print(f"CPU result: {cpu_result}")
    
    # Test FPGA accelerator
    fpga_accelerator = K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier")
    fpga_accelerator.reset_stats()
    
    print(f"\n🚀 Testing K5 FPGA Integration...")
    
    try:
        # Test if matrices are suitable
        suitable = fpga_accelerator.is_fpga_suitable(A, B)
        print(f"FPGA suitable: {suitable}")
        
        if suitable:
            # Test FPGA matrix multiplication
            fpga_result = fpga_accelerator.matrix_multiply_fpga(A, B)
            print(f"FPGA result: {fpga_result}")
            
            # Compare results
            error = torch.mean(torch.abs(cpu_result - fpga_result)).item()
            print(f"Average error: {error:.6f}")
            
            # Get statistics
            stats = fpga_accelerator.get_performance_stats()
            print(f"\n📈 Performance Statistics:")
            print(f"   • FPGA calls: {stats['fpga_calls']}")
            print(f"   • CPU fallbacks: {stats['cpu_fallback_calls']}")
            print(f"   • FPGA usage ratio: {stats['fpga_usage_ratio']:.1%}")
            print(f"   • Has real cycle data: {stats['has_real_cycle_data']}")
            
            if stats['has_real_cycle_data']:
                print(f"   • Real FPGA cycles: {stats['real_fpga_cycles_total']:,}")
                print(f"   • Average cycles: {stats['real_fpga_cycles_average']:,.0f}")
                print("   🎉 Real hardware cycle data available!")
            else:
                print("   📊 Using simulation mode")
            
            if error < 0.1:
                print(f"\n✅ Integration test PASSED - FPGA result matches CPU")
            else:
                print(f"\n⚠️  Integration test WARNING - Large error between FPGA and CPU")
                
        else:
            print("❌ Matrix not suitable for FPGA")
            
    except Exception as e:
        print(f"❌ FPGA integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)


def test_file_interface():
    """Test the file interface directly"""
    print("\n🔧 Testing File Interface")
    print("=" * 30)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temp directory: {temp_dir}")
        
        # Create test matrices
        A = np.array([[1, 2], [3, 4]], dtype=np.int16)
        B = np.array([[5, 6], [7, 8]], dtype=np.int16)
        
        # Create Python interface files manually
        config_file = os.path.join(temp_dir, "matrix_config.txt")
        with open(config_file, 'w') as f:
            f.write("2\n2\n2\n")  # 2x2 @ 2x2
        
        matrix_a_file = os.path.join(temp_dir, "matrix_a.txt")
        with open(matrix_a_file, 'w') as f:
            for value in A.flatten():
                f.write(f"{value}\n")
        
        matrix_b_file = os.path.join(temp_dir, "matrix_b.txt")
        with open(matrix_b_file, 'w') as f:
            for value in B.flatten():
                f.write(f"{value}\n")
        
        print("✅ Created interface files:")
        print(f"   • Config: {config_file}")
        print(f"   • Matrix A: {matrix_a_file}")
        print(f"   • Matrix B: {matrix_b_file}")
        
        # List all files
        files = os.listdir(temp_dir)
        print(f"   • Files in temp dir: {files}")
        
        # Try to call C program (will fail if not compiled, but shows interface)
        c_program_path = "./de10_lite_matrix_multiplier"
        if os.path.isfile(c_program_path):
            print(f"\n📞 Attempting to call C program...")
            import subprocess
            
            try:
                result = subprocess.run(
                    [c_program_path, temp_dir, "cpu_mode"],  # Use CPU mode for testing
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                print(f"Return code: {result.returncode}")
                print(f"Stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"Stderr:\n{result.stderr}")
                    
                # Check for result file
                result_file = os.path.join(temp_dir, "result.txt")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result_data = f.read().strip().split('\n')
                    print(f"✅ Result file created with {len(result_data)} values")
                    print(f"Result values: {result_data}")
                else:
                    print("❌ No result file created")
                    
            except subprocess.TimeoutExpired:
                print("⏰ C program timed out")
            except FileNotFoundError:
                print("❌ C program executable not found")
        else:
            print("⚠️  C program not compiled - skipping execution test")


if __name__ == "__main__":
    test_c_program_integration()
    test_file_interface()