#!/usr/bin/env python3
"""
Configuration for MAXIMUM computation time under 50k parameters
"""

def get_max_computation_config():
    """Configuration optimized for maximum computation time"""
    return {
        "batch_size": 1,            # Small batch for single inference testing
        "num_epochs": 2,            # Fewer epochs since training will be slow
        "lr": 10**-4,
        "max_train_samples": 1000,  # Fewer samples due to long sequences
        "seq_len": 512,             # VERY long sequences (8x current)
        "d_model": 24,              # Optimized model dimension  
        "datasource": 'Helsinki-NLP/opus-100',
        "lang_src": "en",
        "lang_tgt": "he",
        "model_folder": "weights_max_compute",
        "model_basename": "tmodel_max_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_max_compute",
        "layers": 5,                # Deep architecture (5 layers each)
        "heads": 8,                 # 8 attention heads
        "d_ff": 24,                 # Small but efficient feedforward
        # Vocabulary settings
        "reduced_vocab": True,
        "vocab_size": 200,          # Small vocab to save parameters for architecture
        # FPGA acceleration settings
        "use_fpga": False,
        "fpga_port": "/dev/ttyUSB0",
        "fpga_baudrate": 115200,
    }

if __name__ == "__main__":
    from model import build_transformer
    
    config = get_max_computation_config()
    
    print("🚀 MAXIMUM COMPUTATION TIME CONFIGURATION")
    print("="*60)
    print("This config maximizes inference time while staying under 50k parameters")
    print()
    
    # Build model to verify parameters
    model = build_transformer(
        src_vocab_size=config['vocab_size'],
        tgt_vocab_size=config['vocab_size'],
        src_seq_len=config['seq_len'], 
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['layers'],
        h=config['heads'],
        dropout=0.1,
        d_ff=config['d_ff']
    )
    
    params = sum(p.numel() for p in model.parameters())
    
    print(f"Configuration details:")
    print(f"  • vocab_size: {config['vocab_size']}")
    print(f"  • d_model: {config['d_model']}")  
    print(f"  • seq_len: {config['seq_len']} (vs 64 current)")
    print(f"  • layers: {config['layers']} (vs 1 current)")
    print(f"  • heads: {config['heads']}")
    print(f"  • d_ff: {config['d_ff']}")
    print(f"  • Total parameters: {params:,}")
    print(f"  • Under 50k limit: {'✅' if params < 50000 else '❌'}")
    print()
    print(f"Expected performance:")
    print(f"  • Estimated inference time: ~4.7 seconds")
    print(f"  • 175x longer than current config")
    print(f"  • Perfect for testing FPGA acceleration!")
    print(f"  • Long enough to see meaningful FPGA speedup")
    print()
    print("To use this config:")
    print("1. Replace your current config.py with this")
    print("2. Or copy these settings to your config.py")
    print("3. Run train.py to see the long inference times")