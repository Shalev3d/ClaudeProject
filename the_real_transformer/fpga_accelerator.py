import numpy as np
import torch
import torch.nn as nn
import serial
import struct
import time
from typing import Tuple, Optional
import logging

class FPGAAccelerator:
    """
    FPGA Matrix Multiplication Accelerator Interface
    Communicates with DE10-Lite board via UART for matrix operations
    """
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200, 
                 data_width: int = 16, matrix_size: int = 8):
        """
        Initialize FPGA accelerator interface
        
        Args:
            port: Serial port for FPGA communication
            baudrate: Communication baud rate
            data_width: FPGA data width in bits
            matrix_size: Systolic array size
        """
        self.port = port
        self.baudrate = baudrate
        self.data_width = data_width
        self.matrix_size = matrix_size
        self.serial_conn = None
        self.scale_factor = 2**(data_width - 1 - 8)  # Fixed point scaling
        
        # Command codes for FPGA
        self.CMD_MATMUL = 0x01
        self.CMD_RESET = 0x02
        self.CMD_STATUS = 0x03
        
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Establish connection to FPGA"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=30.0,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Test connection
            self._send_reset()
            time.sleep(0.1)
            
            if self._check_status():
                self.logger.info(f"Connected to FPGA on {self.port}")
                return True
            else:
                self.logger.error("FPGA not responding")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to FPGA: {e}")
            return False
    
    def disconnect(self):
        """Close FPGA connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.logger.info("Disconnected from FPGA")
    
    def _send_reset(self):
        """Send reset command to FPGA"""
        cmd = struct.pack('<B', self.CMD_RESET)
        self.serial_conn.write(cmd)
        self.serial_conn.flush()
    
    def _check_status(self) -> bool:
        """Check FPGA status"""
        try:
            cmd = struct.pack('<B', self.CMD_STATUS)
            self.serial_conn.write(cmd)
            self.serial_conn.flush()
            
            response = self.serial_conn.read(1)
            if len(response) == 1:
                status = struct.unpack('<B', response)[0]
                return status == 0x01  # Ready status
            return False
        except:
            return False
    
    def _float_to_fixed(self, value: float) -> int:
        """Convert float to fixed-point representation"""
        return int(np.clip(value * self.scale_factor, 
                          -2**(self.data_width-1), 
                          2**(self.data_width-1)-1))
    
    def _fixed_to_float(self, value: int) -> float:
        """Convert fixed-point to float representation"""
        if value >= 2**(self.data_width-1):
            value -= 2**self.data_width
        return float(value) / self.scale_factor
    
    def _send_matrix(self, matrix: np.ndarray, rows: int, cols: int):
        """Send matrix data to FPGA"""
        # Send matrix dimensions
        dims = struct.pack('<HH', rows, cols)
        self.serial_conn.write(dims)
        
        # Send matrix data in row-major order
        for i in range(rows):
            for j in range(cols):
                fixed_val = self._float_to_fixed(matrix[i, j])
                data = struct.pack('<h', fixed_val)  # signed 16-bit
                self.serial_conn.write(data)
        
        self.serial_conn.flush()
    
    def _receive_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Receive matrix result from FPGA"""
        result = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                data = self.serial_conn.read(2)  # 16-bit result
                if len(data) == 2:
                    fixed_val = struct.unpack('<h', data)[0]
                    result[i, j] = self._fixed_to_float(fixed_val)
                else:
                    raise RuntimeError("Failed to receive complete matrix")
        
        return result
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication using FPGA accelerator
        
        Args:
            A: Input matrix A (M x K)
            B: Input matrix B (K x N)
            
        Returns:
            Result matrix C (M x N)
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            raise RuntimeError("FPGA not connected")
        
        # Validate matrix dimensions
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} x {B.shape}")
        
        M, K = A.shape
        K2, N = B.shape
        
        self.logger.debug(f"FPGA matrix multiply: ({M}x{K}) x ({K2}x{N})")
        
        try:
            # Send command
            cmd = struct.pack('<B', self.CMD_MATMUL)
            self.serial_conn.write(cmd)
            
            # Send matrix A
            self._send_matrix(A, M, K)
            
            # Send matrix B  
            self._send_matrix(B, K, N)
            
            # Wait for computation completion signal
            completion = self.serial_conn.read(1)
            if len(completion) != 1 or struct.unpack('<B', completion)[0] != 0xFF:
                raise RuntimeError("FPGA computation failed")
            
            # Receive result matrix
            result = self._receive_matrix(M, N)
            
            self.logger.debug(f"FPGA computation completed")
            return result
            
        except Exception as e:
            self.logger.error(f"FPGA matrix multiplication failed: {e}")
            raise


class FPGALinear(nn.Module):
    """
    FPGA-accelerated Linear layer replacement for PyTorch
    Drop-in replacement for nn.Linear that uses FPGA for matrix multiplication
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 fpga_accelerator: Optional[FPGAAccelerator] = None):
        super(FPGALinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.fpga_accelerator = fpga_accelerator
        
        # Initialize weights and bias same as nn.Linear
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # Fallback to CPU if FPGA not available
        self.use_fpga = fpga_accelerator is not None and fpga_accelerator.serial_conn
        
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional FPGA acceleration"""
        
        if self.use_fpga and self.fpga_accelerator:
            try:
                # Convert to numpy for FPGA processing
                input_np = input.detach().cpu().numpy()
                weight_np = self.weight.detach().cpu().numpy()
                
                # Reshape input for matrix multiplication
                orig_shape = input_np.shape
                if len(orig_shape) > 2:
                    input_np = input_np.reshape(-1, input_np.shape[-1])
                
                # FPGA matrix multiplication: input @ weight.T
                result_np = self.fpga_accelerator.matrix_multiply(input_np, weight_np.T)
                
                # Convert back to tensor and restore original shape
                result = torch.from_numpy(result_np).to(input.device).to(input.dtype)
                
                if len(orig_shape) > 2:
                    new_shape = list(orig_shape[:-1]) + [self.out_features]
                    result = result.reshape(new_shape)
                
                # Add bias if present
                if self.bias is not None:
                    result = result + self.bias
                
                return result
                
            except Exception as e:
                print(f"FPGA acceleration failed, falling back to CPU: {e}")
                # Fall back to CPU computation
                
        # Standard PyTorch linear layer computation
        return torch.nn.functional.linear(input, self.weight, self.bias)


class FPGAAttention(nn.Module):
    """
    FPGA-accelerated Multi-Head Attention
    Accelerates the key matrix multiplications in attention mechanism
    """
    
    def __init__(self, d_model: int, h: int, dropout: float,
                 fpga_accelerator: Optional[FPGAAccelerator] = None):
        super(FPGAAttention, self).__init__()
        
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h
        
        self.fpga_accelerator = fpga_accelerator
        self.use_fpga = fpga_accelerator is not None
        
        # Use FPGA-accelerated linear layers for Q, K, V, O projections
        self.w_q = FPGALinear(d_model, d_model, bias=False, fpga_accelerator=fpga_accelerator)
        self.w_k = FPGALinear(d_model, d_model, bias=False, fpga_accelerator=fpga_accelerator)
        self.w_v = FPGALinear(d_model, d_model, bias=False, fpga_accelerator=fpga_accelerator)
        self.w_o = FPGALinear(d_model, d_model, bias=False, fpga_accelerator=fpga_accelerator)
        
        self.dropout = nn.Dropout(dropout)
    
    def fpga_attention_matmul(self, query: torch.Tensor, key: torch.Tensor, 
                             value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Compute attention using FPGA for matrix multiplications"""
        
        if not self.use_fpga:
            # Standard attention computation
            d_k = query.shape[-1]
            scores = (query @ key.transpose(-2, -1)) / np.sqrt(d_k)
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attention_weights = torch.softmax(scores, dim=-1)
            if self.dropout:
                attention_weights = self.dropout(attention_weights)
            return attention_weights @ value, attention_weights
        
        try:
            batch_size, num_heads, seq_len, d_k = query.shape
            
            # FPGA acceleration for Q @ K.T
            query_np = query.detach().cpu().numpy()
            key_np = key.detach().cpu().numpy()
            
            # Process each head and batch separately for now
            # (Could be optimized for batch processing)
            scores_list = []
            for b in range(batch_size):
                batch_scores = []
                for h in range(num_heads):
                    q_h = query_np[b, h]  # (seq_len, d_k)
                    k_h = key_np[b, h].T  # (d_k, seq_len)
                    
                    # FPGA matrix multiplication
                    scores_h = self.fpga_accelerator.matrix_multiply(q_h, k_h)
                    scores_h = scores_h / np.sqrt(d_k)
                    batch_scores.append(scores_h)
                scores_list.append(np.stack(batch_scores))
            
            scores = torch.from_numpy(np.stack(scores_list)).to(query.device).to(query.dtype)
            
            # Apply mask and softmax on GPU/CPU
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # FPGA acceleration for attention @ V
            attention_np = attention_weights.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()
            
            output_list = []
            for b in range(batch_size):
                batch_output = []
                for h in range(num_heads):
                    att_h = attention_np[b, h]  # (seq_len, seq_len)
                    v_h = value_np[b, h]        # (seq_len, d_k)
                    
                    # FPGA matrix multiplication
                    output_h = self.fpga_accelerator.matrix_multiply(att_h, v_h)
                    batch_output.append(output_h)
                output_list.append(np.stack(batch_output))
            
            output = torch.from_numpy(np.stack(output_list)).to(query.device).to(query.dtype)
            
            return output, attention_weights
            
        except Exception as e:
            print(f"FPGA attention failed, falling back to CPU: {e}")
            # Fallback to standard computation
            d_k = query.shape[-1]
            scores = (query @ key.transpose(-2, -1)) / np.sqrt(d_k)
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            return attention_weights @ value, attention_weights
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """Forward pass with FPGA-accelerated attention"""
        
        # Project to Q, K, V using FPGA-accelerated linear layers
        query = self.w_q(q)
        key = self.w_k(k)  
        value = self.w_v(v)
        
        # Reshape for multi-head attention
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # FPGA-accelerated attention computation
        x, attention_scores = self.fpga_attention_matmul(query, key, value, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection using FPGA-accelerated linear layer
        return self.w_o(x)


def replace_linear_layers_with_fpga(model: nn.Module, fpga_accelerator: FPGAAccelerator):
    """
    Replace all Linear layers in a model with FPGA-accelerated versions
    
    Args:
        model: PyTorch model to modify
        fpga_accelerator: FPGA accelerator instance
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with FPGA version
            fpga_linear = FPGALinear(
                module.in_features, 
                module.out_features,
                module.bias is not None,
                fpga_accelerator
            )
            
            # Copy weights and bias
            fpga_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                fpga_linear.bias.data = module.bias.data.clone()
            
            setattr(model, name, fpga_linear)
        else:
            # Recursively replace in child modules
            replace_linear_layers_with_fpga(module, fpga_accelerator)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test FPGA accelerator
    fpga = FPGAAccelerator(port='/dev/ttyUSB0')  # Adjust port as needed
    
    if fpga.connect():
        print("FPGA connected successfully!")
        
        # Test matrix multiplication
        A = np.random.randn(4, 6).astype(np.float32)
        B = np.random.randn(6, 8).astype(np.float32)
        
        # CPU reference
        C_cpu = A @ B
        
        # FPGA computation
        C_fpga = fpga.matrix_multiply(A, B)
        
        # Compare results
        error = np.mean(np.abs(C_cpu - C_fpga))
        print(f"Mean absolute error: {error}")
        
        if error < 0.1:  # Tolerance for fixed-point precision
            print("FPGA acceleration working correctly!")
        else:
            print("FPGA results differ significantly from CPU")
        
        fpga.disconnect()
    else:
        print("Failed to connect to FPGA")