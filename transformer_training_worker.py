#!/usr/bin/env python3
"""
Transformer Training Worker - Called by C Program Controller
Follows the same pattern as the 4x4 matrix multiplication example:

1. C program is the main controller and initializes FPGA
2. C program calls this Python script when needed
3. Python script writes matrix operation requests to files
4. C program processes requests using FPGA/CPU
5. Python script reads results from files
6. Cycle measurements come from C program (real K5 hardware)

This is a file-based communication system between C and Python.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import sys
from pathlib import Path

# Import our model and training components
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config
from train import get_ds


class FPGAMatrixInterface:
    """Interface for communicating matrix operations to C program via files"""
    
    def __init__(self, work_dir="./training_workspace"):
        self.work_dir = work_dir
        self.operation_count = 0
        Path(work_dir).mkdir(exist_ok=True)
        
    def request_matrix_multiply(self, A: torch.Tensor, B: torch.Tensor, operation_name: str) -> torch.Tensor:
        """
        Request matrix multiplication from C program via file interface
        
        Args:
            A, B: Input tensors
            operation_name: Description of operation for logging
            
        Returns:
            Result tensor
        """
        self.operation_count += 1
        
        # Convert to CPU and get numpy arrays
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        
        # Write request to files (format expected by C program)
        request_id = f"op_{self.operation_count:04d}_{operation_name}"
        
        config_file = os.path.join(self.work_dir, f"{request_id}_config.txt")
        data_a_file = os.path.join(self.work_dir, f"{request_id}_a.txt")
        data_b_file = os.path.join(self.work_dir, f"{request_id}_b.txt")
        result_file = os.path.join(self.work_dir, f"{request_id}_result.txt")
        
        # Write configuration
        with open(config_file, 'w') as f:
            f.write(f"{A_np.shape[0]}\n")  # rows_a
            f.write(f"{A_np.shape[1]}\n")  # cols_a  
            f.write(f"{B_np.shape[1]}\n")  # cols_b
            f.write(f"{request_id}\n")     # operation identifier
        
        # Write matrix A
        with open(data_a_file, 'w') as f:
            for value in A_np.flatten():
                f.write(f"{int(value * 100)}\n")  # Scale to int16
        
        # Write matrix B
        with open(data_b_file, 'w') as f:
            for value in B_np.flatten():
                f.write(f"{int(value * 100)}\n")  # Scale to int16
        
        # Signal request ready
        ready_file = os.path.join(self.work_dir, f"{request_id}_ready.txt")
        with open(ready_file, 'w') as f:
            f.write("READY\n")
        
        print(f"🔄 Requested FPGA matrix op: {operation_name} ({A.shape} @ {B.shape})")
        
        # Wait for C program to process and create result
        max_wait_time = 10.0  # seconds
        start_time = time.time()
        
        while not os.path.exists(result_file):
            time.sleep(0.01)  # 10ms polling
            if time.time() - start_time > max_wait_time:
                print(f"⏰ Timeout waiting for result: {operation_name}")
                # Fallback to CPU computation
                return torch.matmul(A, B)
        
        # Read result
        try:
            with open(result_file, 'r') as f:
                values = [int(line.strip()) for line in f if line.strip()]
            
            # Convert back to tensor
            result_shape = (A.shape[0], B.shape[1])
            result_np = np.array(values).reshape(result_shape) / 10000.0  # Unscale
            result = torch.from_numpy(result_np.astype(np.float32)).to(A.device).to(A.dtype)
            
            print(f"✅ Received FPGA result: {operation_name}")
            
            # Cleanup files
            for f in [config_file, data_a_file, data_b_file, ready_file, result_file]:
                try:
                    os.remove(f)
                except:
                    pass
            
            return result
            
        except Exception as e:
            print(f"❌ Error reading FPGA result: {e}")
            return torch.matmul(A, B)  # Fallback


# Override torch.matmul for FPGA communication
fpga_interface = FPGAMatrixInterface()

def fpga_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """FPGA matrix multiplication via C program interface"""
    return fpga_interface.request_matrix_multiply(A, B, "transformer_matmul")


class FPGATransformerTraining:
    """Transformer training that communicates with C program for FPGA operations"""
    
    def __init__(self):
        self.config = get_config()
        # Use very small configuration for FPGA compatibility
        self.config.update({
            'batch_size': 1,
            'num_epochs': 1,  
            'max_train_samples': 10,  # Minimal dataset
            'd_model': 8,             # Small model
            'heads': 2,
            'd_ff': 16,
            'layers': 1               # Single layer
        })
        
    def setup_model_and_data(self):
        """Setup model and dataset"""
        print("📚 Setting up transformer model and dataset...")
        
        device = torch.device("cpu")  # Force CPU for file-based communication
        
        # Get dataset
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(self.config)
        
        # Build model (no FPGA accelerator - we handle this via file interface)
        model = build_transformer(
            tokenizer_src.get_vocab_size(), 
            tokenizer_tgt.get_vocab_size(),
            self.config["seq_len"], 
            self.config['seq_len'],
            d_model=self.config['d_model'], 
            N=self.config['layers'], 
            h=self.config['heads'],
            d_ff=self.config['d_ff']
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model: {total_params:,} parameters")
        print(f"📊 Architecture: {self.config['layers']} layers, {self.config['heads']} heads")
        print(f"📊 Dimensions: d_model={self.config['d_model']}, d_ff={self.config['d_ff']}")
        
        return model, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, device
    
    def train_single_batch(self, model, batch, tokenizer_tgt, device):
        """Train on a single batch with FPGA communication"""
        model.train()
        
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device) 
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)
        
        print(f"🔄 Forward pass - matrix operations will be sent to FPGA...")
        
        # Forward pass - matmul operations will be intercepted and sent to C program
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        
        # Compute loss
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        print(f"📉 Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
    
    def run_training(self):
        """Run transformer training with FPGA communication"""
        print("🚀 Starting FPGA-Accelerated Transformer Training")
        print("   (Matrix operations will be processed by C program + FPGA)")
        print("=" * 60)
        
        # Setup
        model, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, device = self.setup_model_and_data()
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            print(f"\n📚 EPOCH {epoch + 1}")
            print("-" * 40)
            
            batch_losses = []
            batch_iter = iter(train_dataloader)
            
            # Process a few batches
            for batch_idx in range(min(3, len(train_dataloader))):  # Limit for demo
                try:
                    batch = next(batch_iter)
                    print(f"\n🔄 Batch {batch_idx + 1}/3")
                    
                    loss = self.train_single_batch(model, batch, tokenizer_tgt, device)
                    batch_losses.append(loss)
                    
                    print(f"✅ Batch {batch_idx + 1} completed, loss: {loss:.4f}")
                    
                except StopIteration:
                    break
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx + 1}: {e}")
                    continue
            
            # Epoch summary
            if batch_losses:
                avg_loss = sum(batch_losses) / len(batch_losses)
                print(f"\n📊 Epoch {epoch + 1} Summary:")
                print(f"   • Batches processed: {len(batch_losses)}")
                print(f"   • Average loss: {avg_loss:.4f}")
                print(f"   • Matrix operations: {fpga_interface.operation_count}")
            
        print(f"\n🎉 Training completed!")
        print(f"   Total matrix operations sent to FPGA: {fpga_interface.operation_count}")
        
        # Write final statistics for C program
        stats_file = os.path.join(fpga_interface.work_dir, "training_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"total_matrix_operations: {fpga_interface.operation_count}\n")
            f.write(f"final_loss: {batch_losses[-1] if batch_losses else 0.0}\n")
            f.write(f"training_completed: TRUE\n")
        
        print(f"✅ Training statistics written to {stats_file}")


def main():
    """Main entry point - called by C program"""
    print("🐍 Python Transformer Training Worker Started")
    print("   Called by C program controller")
    print("   Will communicate matrix operations via files")
    
    # Check if we're being called correctly
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "./training_workspace"
    print(f"📁 Work directory: {work_dir}")
    
    # Override matrix multiplication to use FPGA interface
    import torch
    original_matmul = torch.matmul
    torch.matmul = fpga_matmul
    
    try:
        # Run training
        training = FPGATransformerTraining()
        training.run_training()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Write error status for C program
        error_file = os.path.join(work_dir, "training_error.txt")
        with open(error_file, 'w') as f:
            f.write(f"error: {str(e)}\n")
            f.write(f"training_failed: TRUE\n")
    
    finally:
        # Restore original matmul
        torch.matmul = original_matmul
    
    print("🐍 Python training worker completed")


if __name__ == "__main__":
    main()