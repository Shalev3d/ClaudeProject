#!/usr/bin/env python3
"""
Matrix multiplication result checker for K5 FPGA accelerator
Compares FPGA output against expected mathematical result
"""

import numpy as np
import os
import sys

def check_matrix_result():
    """Verify matrix multiplication results"""
    
    print("="*50)
    print("MATRIX MULTIPLICATION RESULT VERIFICATION")
    print("="*50)
    
    try:
        # Load expected result
        if not os.path.exists('matrix_expected.txt'):
            print("❌ ERROR: matrix_expected.txt not found")
            print("   Run gen_matrix_test.py first to generate test data")
            return False
            
        with open('matrix_expected.txt', 'rb') as f:
            expected_data = f.read()
        expected_result = np.frombuffer(expected_data, dtype=np.int16)
        
        # Load actual result from FPGA
        if not os.path.exists('matrix_test_out.txt'):
            print("❌ ERROR: matrix_test_out.txt not found")
            print("   FPGA computation did not produce output file")
            return False
            
        with open('matrix_test_out.txt', 'rb') as f:
            actual_data = f.read()
        actual_result = np.frombuffer(actual_data, dtype=np.int16)
        
        print(f"Expected result length: {len(expected_result)} elements")
        print(f"Actual result length: {len(actual_result)} elements")
        
        # Check if lengths match
        if len(expected_result) != len(actual_result):
            print("❌ FAIL: Result size mismatch!")
            print(f"   Expected {len(expected_result)} elements")
            print(f"   Got {len(actual_result)} elements")
            return False
        
        # Reshape for matrix comparison (assume square result for simplicity)
        result_dim = int(np.sqrt(len(expected_result)))
        if result_dim * result_dim == len(expected_result):
            expected_matrix = expected_result.reshape(result_dim, result_dim)
            actual_matrix = actual_result.reshape(result_dim, result_dim)
            
            print(f"\nExpected Result Matrix ({result_dim}x{result_dim}):")
            print(expected_matrix)
            print(f"\nActual FPGA Result Matrix ({result_dim}x{result_dim}):")
            print(actual_matrix)
        else:
            print(f"\nExpected Result Vector ({len(expected_result)} elements):")
            print(expected_result)
            print(f"\nActual FPGA Result Vector ({len(actual_result)} elements):")
            print(actual_result)
        
        # Element-wise comparison
        differences = expected_result - actual_result
        max_error = np.max(np.abs(differences))
        num_errors = np.count_nonzero(differences)
        
        print(f"\nComparison Summary:")
        print(f"  Total elements: {len(expected_result)}")
        print(f"  Elements with errors: {num_errors}")
        print(f"  Maximum error: {max_error}")
        
        if num_errors == 0:
            print("\n🎉 PASS: All elements match exactly!")
            print("✅ Matrix multiplication accelerator working correctly")
            return True
        else:
            print(f"\n❌ FAIL: {num_errors} elements don't match")
            print("   Error details:")
            for i, diff in enumerate(differences):
                if diff != 0:
                    print(f"     Element {i}: Expected {expected_result[i]}, Got {actual_result[i]} (diff: {diff})")
                    if i >= 10:  # Limit error output
                        print(f"     ... and {num_errors - 10} more errors")
                        break
            return False
            
    except Exception as e:
        print(f"❌ ERROR during verification: {e}")
        return False

def compare_debug_info():
    """Show debug information from test generation"""
    if os.path.exists('matrix_test_debug.txt'):
        print("\nDebug Information:")
        print("-" * 30)
        with open('matrix_test_debug.txt', 'r') as f:
            print(f.read())

if __name__ == "__main__":
    success = check_matrix_result()
    compare_debug_info()
    
    # Exit with appropriate code for shell scripts
    sys.exit(0 if success else 1)