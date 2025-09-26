"""
K5 FPGA Matrix Multiplication Accelerator
Integrates with K5 processor system for transformer training acceleration
Uses the proven K5 SOC interface for FPGA communication
"""

import torch
import torch.nn as nn
import numpy as np
import subprocess
import os
import struct
import time
from typing import Tuple, Optional
import logging

class K5FPGAAccelerator:
    """
    K5-compatible FPGA Matrix Multiplication Accelerator
    Communicates with FPGA through K5 system like the proven de10_lite_matrix_multiplier
    """
    
    def __init__(self, k5_app_name: str = "de10_lite_matrix_multiplier", 
                 data_width: int = 16, max_matrix_size: int = 8):
        """
        Initialize K5 FPGA accelerator
        
        Args:
            k5_app_name: Name of K5 application for matrix multiplication
            data_width: FPGA data width in bits (16-bit signed integers)
            max_matrix_size: Maximum matrix dimension supported by FPGA
        """
        self.k5_app_name = k5_app_name
        self.data_width = data_width
        self.max_matrix_size = max_matrix_size
        self.scale_factor = 1.0  # No scaling needed for integer operations
        
        # FPGA command codes (matching C implementation)
        self.CMD_MATMUL = 0x01
        self.CMD_STATUS = 0x03
        self.RESP_DONE = 0xFF
        
        self.logger = logging.getLogger(__name__)
        
        # Track performance statistics
        self.fpga_calls = 0
        self.fpga_time_total = 0.0
        self.cpu_fallback_calls = 0
        
        # Wait for FPGA server to be ready
        self._wait_for_fpga_server()
    
    def _wait_for_fpga_server(self):
        """Wait for C program FPGA server to be ready"""
        print(" Waiting for FPGA server to be ready...")
        print("   Make sure the C program is running first: launch_k5_app de10_lite_matrix_multiplier -ccd1 XON")
        
        max_wait_time = 30.0  # seconds
        start_wait = time.time()
        
        # Check multiple possible locations for the ready file
        possible_paths = [
            "fpga_server_ready.txt",
            "./fpga_server_ready.txt", 
            "../fpga_server_ready.txt",
            "threads_runspace/t0/fpga_server_ready.txt",
            "./threads_runspace/t0/fpga_server_ready.txt",
        ]
        
        server_ready = False
        while not server_ready and (time.time() - start_wait) < max_wait_time:
            for path in possible_paths:
                if os.path.exists(path):
                    print(f" Found FPGA server ready file at: {path}")
                    server_ready = True
                    break
            if not server_ready:
                time.sleep(0.5)  # Check every 500ms
        
        if not server_ready:
            print(" FPGA server ready file not found. Continuing without FPGA server...")
            print("   This will run in simulation mode.")
        else:
            print(" FPGA server is ready - Python can now send matrix requests")
        
    def is_fpga_suitable(self, A: torch.Tensor, B: torch.Tensor) -> bool:
        """
        Check if matrices are suitable for FPGA acceleration
        
        Args:
            A, B: Input matrices
            
        Returns:
            True if FPGA should be used, False for CPU fallback
        """
        # Check dimensions
        if len(A.shape) != 2 or len(B.shape) != 2:
            return False
            
        rows_a, cols_a = A.shape
        cols_a_b, cols_b = B.shape
        
        if cols_a != cols_a_b:
            return False
            
        # Check size limits (FPGA is efficient for small-medium matrices)
        if (rows_a > self.max_matrix_size or 
            cols_a > self.max_matrix_size or 
            cols_b > self.max_matrix_size):
            return False
            
        # Check data types (FPGA expects integer-like values)
        if A.dtype not in [torch.float16, torch.float32, torch.int16, torch.int32]:
            return False
            
        return True
    
    def tensor_to_int16(self, tensor: torch.Tensor, scale: float = 100.0) -> np.ndarray:
        """
        Convert PyTorch tensor to int16 for FPGA processing
        
        Args:
            tensor: Input tensor
            scale: Scaling factor for fixed-point conversion
            
        Returns:
            Numpy array of int16 values
        """
        # Scale and convert to int16
        scaled = (tensor.detach().cpu().numpy() * scale).astype(np.int16)
        return scaled
    
    def int16_to_tensor(self, array: np.ndarray, scale: float = 100.0, 
                       device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Convert int16 array back to PyTorch tensor
        
        Args:
            array: Input int16 array
            scale: Scaling factor to reverse
            device: Target device for tensor
            dtype: Target data type
            
        Returns:
            PyTorch tensor
        """
        # Convert back to float and scale down
        float_array = array.astype(np.float32) / scale
        return torch.tensor(float_array, device=device, dtype=dtype)
    
    def create_python_interface_files(self, A: np.ndarray, B: np.ndarray) -> str:
        """
        Create Python interface files for the enhanced C program
        
        Args:
            A, B: Input matrices as int16 numpy arrays
            
        Returns:
            Path to directory containing interface files
        """
        rows_a, cols_a = A.shape
        cols_a_check, cols_b = B.shape
        
        if cols_a != cols_a_check:
            raise ValueError(f"Matrix dimensions don't match: {cols_a} != {cols_a_check}")
        
        # Create temporary directory for Python interface
        temp_dir = "/tmp/k5_fpga_python_interface"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Write configuration file (dimensions only)
        config_file = os.path.join(temp_dir, "matrix_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"{rows_a}\n")
            f.write(f"{cols_a}\n")
            f.write(f"{cols_b}\n")
        
        # Write Matrix A data file (one value per line)
        matrix_a_file = os.path.join(temp_dir, "matrix_a.txt")
        with open(matrix_a_file, 'w') as f:
            for value in A.flatten():
                f.write(f"{value}\n")
        
        # Write Matrix B data file (one value per line)
        matrix_b_file = os.path.join(temp_dir, "matrix_b.txt")
        with open(matrix_b_file, 'w') as f:
            for value in B.flatten():
                f.write(f"{value}\n")
        
        self.logger.debug(f"Created Python interface files in {temp_dir}")
        self.logger.debug(f"  Config: {rows_a}x{cols_a} @ {cols_a}x{cols_b}")
        self.logger.debug(f"  Matrix A: {A.size} elements")
        self.logger.debug(f"  Matrix B: {B.size} elements")
        
        return temp_dir
    
    def run_k5_matrix_multiply(self, temp_dir: str) -> np.ndarray:
        """
        Execute K5 matrix multiplication and retrieve results
        
        Args:
            temp_dir: Directory containing K5 test files
            
        Returns:
            Result matrix as numpy array
        """
        try:
            # Check if we have the actual C program
            c_program_path = "./de10_lite_matrix_multiplier"
            if os.path.isfile(c_program_path):
                # Use real K5 FPGA execution
                return self._run_real_k5_execution(temp_dir, c_program_path)
            else:
                # Fall back to simulation
                self.logger.warning("C program not found, using simulation mode")
                return self._simulate_k5_execution(temp_dir)
                
        except Exception as e:
            raise RuntimeError(f"K5 matrix multiplication failed: {e}")
    
    def _run_real_k5_execution(self, temp_dir: str, c_program_path: str) -> np.ndarray:
        """Execute real K5 FPGA via enhanced C program"""
        import subprocess
        
        try:
            # Execute the enhanced C program with Python interface
            cmd = [c_program_path, temp_dir, "fpga_mode"]
            self.logger.info(f"Executing K5 FPGA with Python interface: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # Increased timeout for real FPGA operations
                cwd=os.getcwd()  # Run from current directory, not temp_dir
            )
            
            if result.returncode != 0:
                self.logger.error(f"K5 execution failed: {result.stderr}")
                raise RuntimeError(f"K5 execution failed: {result.stderr}")
            
            # Parse cycle information from output
            cycles = self._parse_fpga_cycles(result.stdout)
            
            # Read result from Python interface file
            return self._read_python_result(temp_dir)
            
        except subprocess.TimeoutExpired:
            self.logger.error("K5 execution timed out")
            raise RuntimeError("K5 execution timed out")
    
    def _parse_fpga_cycles(self, stdout: str):
        """Parse FPGA cycle information from enhanced C program output"""
        cycles = 0
        lines = stdout.strip().split('\n')
        for line in lines:
            # Look for the specific cycle output format from our C program
            if "FPGA cycles:" in line:
                try:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        cycles = int(parts[1].strip().replace(',', ''))
                        self.logger.info(f"Real FPGA cycles: {cycles}")
                        break
                except ValueError:
                    continue
            elif "CPU cycles:" in line:
                try:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        cycles = int(parts[1].strip().replace(',', ''))
                        self.logger.info(f"CPU cycles: {cycles}")
                        break
                except ValueError:
                    continue
            elif "Measured execution time:" in line and "cycles" in line:
                try:
                    # Extract cycle count from "*** Measured execution time: 12345 K5 effective cycles ***"
                    import re
                    match = re.search(r'(\d+)\s+K5 effective cycles', line)
                    if match:
                        cycles = int(match.group(1))
                        self.logger.info(f"K5 effective cycles: {cycles}")
                        break
                except ValueError:
                    continue
        
        # Store cycle information for statistics
        if cycles > 0:
            self.real_fpga_cycles = getattr(self, 'real_fpga_cycles', 0) + cycles
            
        return cycles
    
    def _read_python_result(self, temp_dir: str) -> np.ndarray:
        """Read result matrix from Python interface file"""
        result_file = os.path.join(temp_dir, "result.txt")
        if not os.path.exists(result_file):
            raise RuntimeError(f"Result file not found: {result_file}")
            
        # Read result values (one per line)
        values = []
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        values.append(int(line))
                    except ValueError:
                        continue
        
        if not values:
            raise RuntimeError("No valid result values found")
            
        # Determine matrix dimensions from values count
        # For now, assume square matrix (this could be improved)
        total_elements = len(values)
        side_length = int(total_elements ** 0.5)
        
        if side_length * side_length == total_elements:
            # Square matrix
            result = np.array(values, dtype=np.int16).reshape(side_length, side_length)
        else:
            # Non-square - try to infer dimensions (improvement needed)
            # For now, return as 1D and let caller handle reshaping
            result = np.array(values, dtype=np.int16).reshape(-1, 1)
            self.logger.warning(f"Non-square result matrix inferred: {result.shape}")
        
        self.logger.debug(f"Read {len(values)} result values, shaped as {result.shape}")
        return result
    
    def _read_k5_result(self, temp_dir: str) -> np.ndarray:
        """Read result from K5 output file"""
        output_file = os.path.join(temp_dir, "matrix_test_out.txt")
        if not os.path.exists(output_file):
            # Fall back to simulation
            return self._simulate_k5_execution(temp_dir)
            
        # Read hex format result file
        with open(output_file, 'r') as f:
            hex_data = f.read().replace('\n', ' ').replace(' ', '')
        
        # Parse hex data back to matrix
        result_bytes = bytes.fromhex(hex_data)
        result_data = np.frombuffer(result_bytes, dtype=np.int16)
        
        # Determine matrix dimensions (this needs to be improved)
        # For now, assume square matrix
        size = int(np.sqrt(len(result_data)))
        if size * size == len(result_data):
            return result_data.reshape(size, size)
        else:
            # Fall back to simulation if we can't parse
            return self._simulate_k5_execution(temp_dir)
    
    def _simulate_k5_execution(self, temp_dir: str) -> np.ndarray:
        """
        Simulate K5 execution for testing purposes
        In production, this would be replaced with actual K5 integration
        """
        # Read the input configuration to determine matrix dimensions
        config_file = os.path.join(temp_dir, "matrix_test_config.txt")
        with open(config_file, 'r') as f:
            hex_data = f.read().replace('\n', ' ').replace(' ', '')
        
        # Parse dimensions from config (first 12 bytes = 3 int32s)
        config_bytes = bytes.fromhex(hex_data[:24])  # First 12 bytes
        rows_a, cols_a, cols_b = struct.unpack('<III', config_bytes)
        
        # Read matrix data
        input_file = os.path.join(temp_dir, "matrix_test_in.txt") 
        with open(input_file, 'r') as f:
            hex_data = f.read().replace('\n', ' ').replace(' ', '')
        
        matrix_bytes = bytes.fromhex(hex_data)
        matrix_data = np.frombuffer(matrix_bytes, dtype=np.int16)
        
        # Split into A and B matrices
        a_size = rows_a * cols_a
        A = matrix_data[:a_size].reshape(rows_a, cols_a)
        B = matrix_data[a_size:].reshape(cols_a, cols_b)
        
        # Perform matrix multiplication
        C = np.matmul(A, B).astype(np.int16)
        
        # Write result file (simulating K5 output)
        output_file = os.path.join(temp_dir, "matrix_test_out.txt")
        result_bytes = C.tobytes()
        with open(output_file, 'w') as f:
            hex_bytes = [f"{byte:02x}" for byte in result_bytes]
            for i in range(0, len(hex_bytes), 16):
                line = ' '.join(hex_bytes[i:i+16])
                f.write(line + '\n')
        
        return C
    
    def matrix_multiply_fpga(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication using K5 FPGA accelerator via C program communication
        
        Args:
            A: Left matrix (M x K)
            B: Right matrix (K x N)
            
        Returns:
            Result matrix (M x N) computed on FPGA via C program
        """
        if not self.is_fpga_suitable(A, B):
            self.logger.debug("Using CPU fallback for matrix multiplication")
            self.cpu_fallback_calls += 1
            return torch.matmul(A, B)
        
        start_time = time.time()
        
        try:
            # Send matrix operation request to C program via file interface
            result = self._send_matrix_to_c_program(A, B)
            
            # Update statistics
            self.fpga_calls += 1
            self.fpga_time_total += time.time() - start_time
            
            self.logger.debug(f"FPGA matrix multiply via C program: {A.shape} @ {B.shape} = {result.shape}")
            return result
            
        except Exception as e:
            self.logger.warning(f"FPGA matrix multiplication failed, falling back to CPU: {e}")
            self.cpu_fallback_calls += 1
            return torch.matmul(A, B)
    
    def _send_matrix_to_c_program(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Send matrix operation to C program via file interface
        This communicates with the C program running on K5 server
        """
        # Convert tensors to numpy and scale for FPGA (int16)
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        
        # Scale to int16 range for FPGA processing
        scale_factor = 100
        A_scaled = (A_np * scale_factor).astype(np.int16)
        B_scaled = (B_np * scale_factor).astype(np.int16)
        
        # Generate unique request ID
        request_id = self.fpga_calls
        
        # Try to write files in the same directory as the C program
        # Check if we're in threads_runspace directory structure
        base_path = ""
        if os.path.exists("threads_runspace/t0/"):
            base_path = "threads_runspace/t0/"
        elif os.path.exists("./t0/"):
            base_path = "./t0/"
        
        # File names that C program will monitor for
        config_file = f"{base_path}fpga_matrix_request_{request_id}_config.txt"
        data_a_file = f"{base_path}fpga_matrix_request_{request_id}_data_a.txt"
        data_b_file = f"{base_path}fpga_matrix_request_{request_id}_data_b.txt"
        result_file = f"{base_path}fpga_matrix_request_{request_id}_result.txt"
        
        try:
            # Write configuration file
            with open(config_file, 'w') as f:
                f.write(f"{A_scaled.shape[0]}\n")  # rows_a
                f.write(f"{A_scaled.shape[1]}\n")  # cols_a
                f.write(f"{B_scaled.shape[1]}\n")  # cols_b
            
            # Write matrix A data
            with open(data_a_file, 'w') as f:
                for value in A_scaled.flatten():
                    f.write(f"{value}\n")
            
            # Write matrix B data
            with open(data_b_file, 'w') as f:
                for value in B_scaled.flatten():
                    f.write(f"{value}\n")
            
            print(f" Sent matrix {A.shape} @ {B.shape} request #{request_id} to C program")
            print(f" Files created at: {base_path if base_path else 'current directory'}")
            print(f"   Config: {config_file}")
            print(f"   Result: {result_file}")
            
            # Wait for C program to process and create result file
            max_wait_time = 30.0  # seconds
            start_wait = time.time()
            
            while not os.path.exists(result_file):
                time.sleep(0.01)  # 10ms polling
                if time.time() - start_wait > max_wait_time:
                    raise RuntimeError(f"Timeout waiting for C program result {request_id}")
            
            # Read result from C program
            with open(result_file, 'r') as f:
                result_values = []
                for line in f:
                    line = line.strip()
                    if line:
                        result_values.append(int(line))
            
            # Convert back to tensor
            result_shape = (A.shape[0], B.shape[1])
            result_np = np.array(result_values).reshape(result_shape)
            
            # Unscale from int16 back to float
            result_np = result_np.astype(np.float32) / (scale_factor * scale_factor)
            
            result_tensor = torch.from_numpy(result_np).to(A.device).to(A.dtype)
            
            print(f" Received FPGA result for request #{request_id} from C program")
            
            # Clean up result file (C program cleans up request files)
            try:
                os.remove(result_file)
            except:
                pass
                
            return result_tensor
            
        except Exception as e:
            # Clean up files on error
            for filename in [config_file, data_a_file, data_b_file, result_file]:
                try:
                    os.remove(filename)
                except:
                    pass
            raise e
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics including real FPGA cycles"""
        avg_time = self.fpga_time_total / max(self.fpga_calls, 1)
        real_cycles = getattr(self, 'real_fpga_cycles', 0)
        avg_cycles = real_cycles / max(self.fpga_calls, 1)
        
        return {
            'fpga_calls': self.fpga_calls,
            'cpu_fallback_calls': self.cpu_fallback_calls,
            'total_calls': self.fpga_calls + self.cpu_fallback_calls,
            'fpga_time_total': self.fpga_time_total,
            'fpga_time_average': avg_time,
            'fpga_usage_ratio': self.fpga_calls / max(self.fpga_calls + self.cpu_fallback_calls, 1),
            'real_fpga_cycles_total': real_cycles,
            'real_fpga_cycles_average': avg_cycles,
            'has_real_cycle_data': real_cycles > 0
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.fpga_calls = 0
        self.fpga_time_total = 0.0
        self.cpu_fallback_calls = 0
        self.real_fpga_cycles = 0


# Global instance for transformer integration
k5_fpga_accelerator = K5FPGAAccelerator()


def fpga_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Global function for FPGA matrix multiplication
    This replaces torch.matmul calls in the transformer
    
    Args:
        A: Left matrix
        B: Right matrix
        
    Returns:
        Result matrix
    """
    return k5_fpga_accelerator.matrix_multiply_fpga(A, B)