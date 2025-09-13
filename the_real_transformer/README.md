# The Real Transformer

This folder contains the essential files for the complete transformer implementation with FPGA acceleration support.

## 📁 Folder Contents

### Python Files (Core Implementation)

- **`config.py`** - Main configuration file with optimized settings
  - vocab_size: 350 (optimized for maximum computation)
  - seq_len: 1024 (ultra-long sequences)
  - layers: 6 (deepest architecture under 50k parameters)
  - d_model: 16, heads: 4, d_ff: 16

- **`model.py`** - Complete transformer architecture
  - Encoder-decoder transformer implementation
  - Multi-head attention with FPGA acceleration support
  - Feed-forward networks with optional FPGA acceleration
  - Layer normalization and residual connections

- **`train.py`** - Main training script
  - Complete training loop with epoch-by-epoch cycle analysis
  - FPGA vs CPU performance comparison after each epoch
  - Automatic tokenizer generation based on config
  - TensorBoard logging and model saving

- **`dataset.py`** - Data loading and preprocessing
  - Bilingual dataset handling
  - Tokenization and sequence padding
  - Causal masking for decoder

- **`cycle_counter.py`** - Performance analysis and cycle counting
  - Detailed CPU cycle estimation
  - FPGA performance projections
  - Component-wise profiling (encoder, decoder, projection)

- **`fpga_accelerator.py`** - FPGA interface and acceleration
  - UART communication with FPGA board
  - Matrix multiplication acceleration
  - Attention computation offloading
  - Hardware-software interface

### FPGA Files (Hardware Implementation)

- **`fpga/de10_lite_top.sv`** - Top-level FPGA module for DE10-Lite board
- **`fpga/host_interface.sv`** - Host communication interface
- **`fpga/matrix_multiplier.sv`** - Hardware matrix multiplication unit
- **`fpga/uart_controller.sv`** - UART communication controller
- **`fpga/pll_100mhz.sv`** - Phase-locked loop for 100MHz clock
- **`fpga/testbench.sv`** - Hardware testbench for verification

## 🚀 How to Run

### Prerequisites
```bash
pip install torch datasets tokenizers tqdm tensorboard torchmetrics
```

### Basic Usage
```bash
python train.py
```

### Expected Output
- Model: ~43,134 parameters (under 50k limit)
- Inference time: ~140ms (maximum achievable under parameter limit)
- FPGA analysis after each epoch
- Automatic tokenizer generation for vocab_size=350

## ⚙️ Configuration

The transformer is optimized for:
- **Maximum computation time** under 50k parameters
- **FPGA acceleration readiness** with parallel attention heads
- **Long sequence processing** (1024 tokens)
- **Deep architecture** (6 encoder + 6 decoder layers)

## 🔧 FPGA Deployment

The SystemVerilog files in `fpga/` directory can be synthesized and deployed to:
- Intel DE10-Lite FPGA board
- 100MHz clock frequency
- UART communication at 115200 baud rate

## 📊 Performance Analysis

After each epoch, you'll see:
- Single inference cycle count and timing
- Full epoch time estimates
- FPGA vs CPU performance comparison
- Speedup potential analysis

## 📝 Notes

- Tokenizer files are generated automatically based on config settings
- Model weights are saved in `weights_max_compute/` folder
- TensorBoard logs are saved in `runs/tmodel_max_compute/`
- All parameters are optimized for the longest inference time possible under 50k parameter limit