#!/usr/bin/env python3
"""
Test different transformer configurations and compare parameter counts
"""

import torch
import torch.nn as nn
from model import build_transformer
from config import get_config

def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_configuration(config_name, config_params):
    """Test a specific configuration and return parameter count"""
    print(f"\n{'='*60}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*60}")
    
    # Print configuration details
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    
    # Build the transformer with this configuration
    try:
        transformer = build_transformer(
            src_vocab_size=config_params['vocab_size'],
            tgt_vocab_size=config_params['vocab_size'], 
            src_seq_len=config_params['seq_len'],
            tgt_seq_len=config_params['seq_len'],
            d_model=config_params['d_model'],
            N=config_params['layers'],
            h=config_params['heads'],
            dropout=0.1,
            d_ff=config_params['d_ff']
        )
        
        param_count = count_parameters(transformer)
        print(f"\n  Total Parameters: {param_count:,}")
        
        # Calculate model size in MB (assuming float32)
        model_size_mb = param_count * 4 / (1024 * 1024)
        print(f"  Model Size: {model_size_mb:.2f} MB")
        
        return param_count, model_size_mb
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, 0

def main():
    print("Transformer Configuration Parameter Analysis")
    print("="*80)
    
    # Get default configuration
    default_config = get_config()
    
    # Test configurations with different parameter scales
    configurations = {
        "Tiny (Current Config)": {
            "vocab_size": default_config['vocab_size'],
            "d_model": default_config['d_model'],
            "seq_len": default_config['seq_len'],
            "layers": default_config['layers'],
            "heads": default_config['heads'],
            "d_ff": default_config['d_ff']
        },
        
        "Ultra-Tiny": {
            "vocab_size": 200,
            "d_model": 32,
            "seq_len": 16,
            "layers": 1,
            "heads": 2,
            "d_ff": 32
        },
        
        "Small": {
            "vocab_size": 2000,
            "d_model": 64,
            "seq_len": 32,
            "layers": 2,
            "heads": 4,
            "d_ff": 128
        },
        
        "Medium": {
            "vocab_size": 20000,
            "d_model": 128,
            "seq_len": 64,
            "layers": 4,
            "heads": 8,
            "d_ff": 256
        },
        
        "Large Vocab": {
            "vocab_size": 100000,
            "d_model": 64,
            "seq_len": 32,
            "layers": 1,
            "heads": 4,
            "d_ff": 128
        },
        
        "Deep Narrow": {
            "vocab_size": 2000,
            "d_model": 48,
            "seq_len": 32,
            "layers": 8,
            "heads": 3,
            "d_ff": 48
        },
        
        "Wide Shallow": {
            "vocab_size": 2000,
            "d_model": 256,
            "seq_len": 32,
            "layers": 1,
            "heads": 8,
            "d_ff": 512
        }
    }
    
    results = {}
    
    # Test each configuration
    for name, config in configurations.items():
        param_count, size_mb = test_configuration(name, config)
        results[name] = {
            'params': param_count,
            'size_mb': size_mb,
            'config': config
        }
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<20} {'Parameters':<15} {'Size (MB)':<10} {'Key Differences'}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['params'])
    
    for name, data in sorted_results:
        config = data['config']
        key_info = f"d={config['d_model']}, L={config['layers']}, H={config['heads']}, V={config['vocab_size']}"
        print(f"{name:<20} {data['params']:>13,} {data['size_mb']:>9.1f} {key_info}")
    
    # Analysis of parameter scaling
    print(f"\n{'='*80}")
    print("PARAMETER SCALING ANALYSIS")
    print(f"{'='*80}")
    
    print("\nHow different components affect parameter count:")
    
    # Vocabulary size effect
    tiny_params = results["Ultra-Tiny"]["params"]
    large_vocab_params = results["Large Vocab"]["params"] 
    print(f"\n1. VOCABULARY SIZE IMPACT:")
    print(f"   Ultra-Tiny (vocab=200): {tiny_params:,} parameters")
    print(f"   Large Vocab (vocab=100k): {large_vocab_params:,} parameters")
    print(f"   Vocabulary scaling factor: {large_vocab_params / tiny_params:.1f}x")
    print(f"   → Vocabulary size has huge impact due to embedding layers")
    
    # Model depth effect  
    small_params = results["Small"]["params"]
    deep_params = results["Deep Narrow"]["params"]
    print(f"\n2. MODEL DEPTH IMPACT:")
    print(f"   Small (2 layers): {small_params:,} parameters")
    print(f"   Deep Narrow (8 layers): {deep_params:,} parameters") 
    print(f"   Depth scaling factor: {deep_params / small_params:.1f}x")
    print(f"   → More layers = linear increase in parameters")
    
    # Model width effect
    wide_params = results["Wide Shallow"]["params"]
    print(f"\n3. MODEL WIDTH IMPACT:")
    print(f"   Small (d_model=64): {small_params:,} parameters")
    print(f"   Wide Shallow (d_model=256): {wide_params:,} parameters")
    print(f"   Width scaling factor: {wide_params / small_params:.1f}x")
    print(f"   → Model width has quadratic effect (attention matrices scale as d²)")

if __name__ == "__main__":
    main()