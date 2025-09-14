#!/usr/bin/env python3
"""
Integration layer that uses the actual de10_lite_matrix_multiplier.c program
for real FPGA cycle measurements
"""

import subprocess
import os
import tempfile
import struct
import numpy as np
import torch
import time
from typing import Tuple, Optional


class RealFPGAIntegration:
    """
    Integration with the actual K5 C program for real FPGA cycle measurements
    Uses the proven de10_lite_matrix_multiplier.c program
    """
    
    def __init__(self, k5_executable_path: str = "./de10_lite_matrix_multiplier"):
        """
        Initialize with path to the compiled C program
        
        Args:
            k5_executable_path: Path to compiled de10_lite_matrix_multiplier executable
        """
        self.k5_executable_path = k5_executable_path
        self.max_matrix_size = 8  # FPGA systolic array size
        
        # Performance tracking
        self.fpga_calls = 0
        self.cpu_fallback_calls = 0
        self.fpga_cycles_total = 0
        self.cpu_cycles_total = 0
        self.fpga_time_total = 0.0
        
    def is_executable_available(self) -> bool:
        """Check if the K5 executable is available"""
        return os.path.isfile(self.k5_executable_path)
    
    def is_matrix_suitable(self, A: torch.Tensor, B: torch.Tensor) -> bool:
        """Check if matrices can be processed by FPGA"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            return False
            
        rows_a, cols_a = A.shape
        cols_b = B.shape[1]
        
        # Check size limits (8x8 systolic array)
        if (rows_a > self.max_matrix_size or 
            cols_a > self.max_matrix_size or 
            cols_b > self.max_matrix_size):
            return False
            
        return True
    
    def create_c_program_input(self, A: np.ndarray, B: np.ndarray, temp_dir: str) -> str:
        """
        Create input files for the C program in the format it expects
        
        Args:
            A, B: Input matrices (already in int16 format)
            temp_dir: Temporary directory for files
            
        Returns:
            Path to input configuration file
        """
        rows_a, cols_a = A.shape
        cols_b = B.shape[1]
        
        # Create configuration file that the C program expects
        config_file = os.path.join(temp_dir, "matrix_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"{rows_a}\\n")  # Matrix A rows
            f.write(f"{cols_a}\\n")  # Matrix A cols / Matrix B rows  
            f.write(f"{cols_b}\\n")  # Matrix B cols
            
        # Create matrix data files
        matrix_a_file = os.path.join(temp_dir, "matrix_a.txt")
        matrix_b_file = os.path.join(temp_dir, "matrix_b.txt")
        
        # Write Matrix A
        with open(matrix_a_file, 'w') as f:
            for i in range(rows_a):
                for j in range(cols_a):
                    f.write(f"{A[i,j]}\\n")
        
        # Write Matrix B  
        with open(matrix_b_file, 'w') as f:
            for i in range(cols_a):  # cols_a == rows_b
                for j in range(cols_b):
                    f.write(f"{B[i,j]}\\n")
                    
        return config_file
    
    def run_fpga_matrix_multiply(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Run matrix multiplication using the real FPGA via C program
        
        Args:
            A: Left matrix (M x K)  
            B: Right matrix (K x N)
            
        Returns:
            Tuple of (result_tensor, cycle_info)
        """
        
        if not self.is_executable_available():
            raise RuntimeError(f"K5 executable not found at {self.k5_executable_path}")
            
        if not self.is_matrix_suitable(A, B):
            self.cpu_fallback_calls += 1
            return torch.matmul(A, B), {'source': 'cpu_fallback', 'cycles': 0}
        
        # Convert to int16 for FPGA processing
        scale = 100.0
        A_int16 = (A.detach().cpu().numpy() * scale).astype(np.int16)
        B_int16 = (B.detach().cpu().numpy() * scale).astype(np.int16)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create input files
                config_file = self.create_c_program_input(A_int16, B_int16, temp_dir)
                
                # Run the C program
                start_time = time.perf_counter()
                
                cmd = [
                    self.k5_executable_path,
                    temp_dir,  # Pass temp directory as argument
                    "fpga_mode"  # Force FPGA mode
                ]
                
                print(f"🚀 Executing K5 FPGA: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                end_time = time.perf_counter()
                total_time = end_time - start_time
                
                if result.returncode != 0:
                    print(f"⚠️  C program failed: {result.stderr}")
                    self.cpu_fallback_calls += 1
                    return torch.matmul(A, B), {'source': 'cpu_fallback_error', 'cycles': 0}
                
                # Parse output for cycle information
                cycles_info = self.parse_c_program_output(result.stdout)
                
                # Read result matrix (assuming C program writes to result.txt)
                result_file = os.path.join(temp_dir, "result.txt")
                if not os.path.exists(result_file):
                    print(f"⚠️  Result file not found, using CPU fallback")
                    self.cpu_fallback_calls += 1
                    return torch.matmul(A, B), {'source': 'cpu_fallback_no_result', 'cycles': 0}
                
                # Read and parse result matrix
                C_int16 = self.read_result_matrix(result_file, A.shape[0], B.shape[1])
                
                # Convert back to float tensor
                C_float = C_int16.astype(np.float32) / (scale * scale)
                result_tensor = torch.from_numpy(C_float).to(A.device).to(A.dtype)
                
                # Update statistics
                self.fpga_calls += 1
                self.fpga_time_total += total_time
                
                if 'fpga_cycles' in cycles_info:
                    self.fpga_cycles_total += cycles_info['fpga_cycles']
                
                cycles_info.update({
                    'source': 'fpga',
                    'total_time': total_time,
                    'scale_factor': scale
                })
                
                return result_tensor, cycles_info
                
            except subprocess.TimeoutExpired:
                print("⚠️  C program timed out, using CPU fallback")
                self.cpu_fallback_calls += 1
                return torch.matmul(A, B), {'source': 'cpu_fallback_timeout', 'cycles': 0}
            
            except Exception as e:
                print(f"⚠️  FPGA execution failed: {e}, using CPU fallback") 
                self.cpu_fallback_calls += 1
                return torch.matmul(A, B), {'source': 'cpu_fallback_exception', 'cycles': 0}
    
    def parse_c_program_output(self, stdout: str) -> dict:
        """
        Parse the C program output to extract cycle information
        
        Args:
            stdout: Standard output from the C program
            
        Returns:
            Dictionary with cycle information
        """
        cycles_info = {}
        
        lines = stdout.strip().split('\\n')
        for line in lines:
            # Look for cycle information in output
            if "cycles:" in line.lower():
                try:
                    # Extract number after "cycles:"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        cycles = int(parts[1].strip().replace(',', ''))
                        if "fpga" in line.lower():
                            cycles_info['fpga_cycles'] = cycles
                        elif "software" in line.lower() or "cpu" in line.lower():
                            cycles_info['cpu_cycles'] = cycles
                        else:
                            cycles_info['total_cycles'] = cycles
                except ValueError:
                    continue
            
            # Look for timing information
            if "time:" in line.lower() and "ms" in line.lower():
                try:
                    # Extract time in milliseconds
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "time:" and i + 1 < len(parts):
                            time_str = parts[i + 1].replace("ms", "").replace(",", "")
                            cycles_info['reported_time_ms'] = float(time_str)
                            break
                except ValueError:
                    continue
        
        return cycles_info
    
    def read_result_matrix(self, result_file: str, rows: int, cols: int) -> np.ndarray:
        """Read result matrix from C program output file"""
        result = np.zeros((rows, cols), dtype=np.int16)
        
        with open(result_file, 'r') as f:
            values = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        values.append(int(line))
                    except ValueError:
                        continue
        
        # Reshape to matrix
        if len(values) >= rows * cols:
            result = np.array(values[:rows * cols], dtype=np.int16).reshape(rows, cols)
        
        return result
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics including real FPGA cycles"""
        avg_fpga_time = self.fpga_time_total / max(self.fpga_calls, 1)
        avg_fpga_cycles = self.fpga_cycles_total / max(self.fpga_calls, 1)
        
        return {
            'fpga_calls': self.fpga_calls,
            'cpu_fallback_calls': self.cpu_fallback_calls,
            'total_calls': self.fpga_calls + self.cpu_fallback_calls,
            'fpga_time_total': self.fpga_time_total,
            'fpga_time_average': avg_fpga_time,
            'fpga_cycles_total': self.fpga_cycles_total,
            'fpga_cycles_average': avg_fpga_cycles,
            'fpga_usage_ratio': self.fpga_calls / max(self.fpga_calls + self.cpu_fallback_calls, 1)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.fpga_calls = 0
        self.cpu_fallback_calls = 0
        self.fpga_cycles_total = 0
        self.cpu_cycles_total = 0
        self.fpga_time_total = 0.0


# Integration function for transformer
def fpga_matmul_with_real_cycles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using real FPGA with cycle counting
    This replaces the simulated version
    """
    global real_fpga_accelerator
    
    if 'real_fpga_accelerator' not in globals():
        # Initialize on first use
        real_fpga_accelerator = RealFPGAIntegration()
    
    result, cycles_info = real_fpga_accelerator.run_fpga_matrix_multiply(A, B)
    
    # Log cycle information
    if cycles_info.get('source') == 'fpga' and 'fpga_cycles' in cycles_info:
        print(f"🔢 FPGA matrix multiply: {A.shape} @ {B.shape} = {cycles_info['fpga_cycles']} cycles")
    
    return result


if __name__ == "__main__":
    print("🔗 Real FPGA Integration")
    print("This module integrates with de10_lite_matrix_multiplier.c for real cycle measurements")
    print()
    print("To use:")
    print("1. Compile your C program: gcc -o de10_lite_matrix_multiplier de10_lite_matrix_multiplier.c")
    print("2. Replace fpga_matmul imports with fpga_matmul_with_real_cycles")
    print("3. Run transformer with real FPGA cycle measurements")