# Setup Instructions for Friends

## 📋 Prerequisites

### 1. Install Python (3.8 or higher)
```bash
# Check if Python is installed
python --version
# or
python3 --version
```

### 2. Install Required Python Packages
```bash
pip install torch datasets tokenizers tqdm tensorboard torchmetrics
```

**Alternative with conda:**
```bash
conda install pytorch datasets tokenizers tqdm tensorboard torchmetrics -c pytorch -c conda-forge
```

### 3. Install Additional Dependencies (if needed)
```bash
pip install transformers huggingface_hub
```

## 🚀 How to Run

### Option 1: Quick Start (Recommended)
```bash
cd the_real_transformer
python train.py
```

### Option 2: Step by Step
1. **Navigate to the folder:**
   ```bash
   cd the_real_transformer
   ```

2. **Check configuration (optional):**
   ```bash
   python -c "from config import get_config; print(get_config())"
   ```

3. **Run training:**
   ```bash
   python train.py
   ```

## 📊 What to Expect

### During First Run:
1. **Dataset Download**: The system will automatically download the Helsinki-NLP translation dataset
2. **Tokenizer Creation**: Will build tokenizers for English→Hebrew translation with 350 vocabulary size
3. **Model Creation**: Creates a 43,134 parameter transformer
4. **Training Start**: Begins training with cycle analysis after each epoch

### Expected Output:
```
🏗️  Model Information:
   • Source vocabulary: ~350 tokens
   • Target vocabulary: ~350 tokens
   • Total parameters: 43,134
   • Trainable parameters: 43,134
   • Model size: 0.2 MB
   • Training samples: 1,000
   • Device: cpu

⏱️  Performance Analysis:
   • Inference time: ~140.0 ms
   • Estimated cycles: ~400,000,000
   • FPGA speedup potential: X.XXx

Processing Epoch 01: 100%|████████| 1000/1000 [XX:XX<00:00, X.XXit/s]

⏱️  EPOCH 1 CYCLE ANALYSIS
==================================================
Single inference:
  • Time: 140.0 ms
  • Cycles: 420,000,000
  • Sequence length: 1024
  • Cycles per token: 410,156

Full epoch estimate (100 batches):
  • Total time: 14.0 seconds
  • Total cycles: 42,000,000,000
  • Minutes: 0.2

FPGA estimates (100MHz):
  • Single inference FPGA time: 4200.0 ms
  • Single inference speedup: 0.03x
  • Full epoch FPGA time: 420.0 seconds
  • Full epoch speedup: 0.03x
  💻 CPU remains faster for this model size
==================================================
```

## 🛠️ Troubleshooting

### Common Issues:

**1. "No module named 'torch'"**
```bash
pip install torch
```

**2. "No module named 'datasets'"**
```bash
pip install datasets
```

**3. "CUDA not available" (this is normal)**
- The model will run on CPU, which is expected
- Performance analysis is optimized for CPU

**4. "Download failed"**
```bash
# Check internet connection
# Try running with smaller dataset:
# Edit config.py and change max_train_samples to 100
```

**5. "Permission denied"**
```bash
# Make sure you have write permissions in the folder
chmod +x train.py
```

## ⚙️ Customization (Optional)

### Change Training Settings:
Edit `config.py` to modify:
```python
"num_epochs": 2,        # Number of training epochs
"max_train_samples": 1000,  # Number of training samples
"batch_size": 1,        # Batch size
```

### Change Model Size:
```python
"vocab_size": 350,      # Vocabulary size
"seq_len": 1024,        # Sequence length
"layers": 6,            # Number of layers
```

## 📁 Generated Files

After running, you'll see these new folders/files:
- `weights_max_compute/` - Saved model checkpoints
- `runs/tmodel_max_compute/` - TensorBoard logs
- `tokenizer_en.json` - English tokenizer (auto-generated)
- `tokenizer_he.json` - Hebrew tokenizer (auto-generated)

## 🔧 Hardware Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 1GB free disk space
- Internet connection (first run only)

**Recommended:**
- Python 3.9+
- 8GB RAM
- Multi-core CPU for faster training

## 💡 Tips

1. **First run takes longer** due to dataset download and tokenizer creation
2. **Subsequent runs are faster** as everything is cached
3. **Monitor progress** - each epoch shows detailed cycle analysis
4. **Stop anytime** with Ctrl+C - model saves after each epoch
5. **View logs** with: `tensorboard --logdir runs/tmodel_max_compute`

## 🆘 Need Help?

If something doesn't work:
1. Check you have all prerequisites installed
2. Make sure you have internet connection for first run
3. Check you have write permissions in the folder
4. Try running with fewer training samples by editing config.py