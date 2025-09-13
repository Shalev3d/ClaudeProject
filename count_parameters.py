#!/usr/bin/env python3
"""
Calculate transformer model parameters for different configurations
"""

def calculate_transformer_parameters(vocab_size, d_model, seq_len, n_layers, n_heads, d_ff):
    """
    Calculate the number of parameters in a transformer model
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        seq_len: Sequence length
        n_layers: Number of layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
    """
    
    print(f"Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Feedforward dimension: {d_ff}")
    print()
    
    # Input embeddings (source and target)
    src_embedding = vocab_size * d_model
    tgt_embedding = vocab_size * d_model
    total_embeddings = src_embedding + tgt_embedding
    
    # Positional encodings (usually not learned parameters, but let's count them)
    src_pos = seq_len * d_model 
    tgt_pos = seq_len * d_model
    total_pos = src_pos + tgt_pos
    
    # Multi-head attention parameters per layer
    # Q, K, V projections: 3 * d_model * d_model
    # Output projection: d_model * d_model
    attention_per_layer = 4 * d_model * d_model
    
    # Feedforward network per layer
    # Linear 1: d_model * d_ff
    # Linear 2: d_ff * d_model  
    ff_per_layer = d_model * d_ff + d_ff * d_model
    
    # Layer normalization per layer (2 per layer: after attention and after FF)
    # Each LayerNorm has 2 * d_model parameters (gamma and beta)
    ln_per_layer = 2 * 2 * d_model
    
    # Total per layer (for both encoder and decoder layers)
    params_per_layer = attention_per_layer + ff_per_layer + ln_per_layer
    total_layers_params = params_per_layer * n_layers * 2  # encoder + decoder
    
    # Output projection layer
    output_projection = d_model * vocab_size
    
    # Total parameters
    total_params = total_embeddings + total_pos + total_layers_params + output_projection
    
    print(f"Parameter breakdown:")
    print(f"  Source embeddings: {src_embedding:,}")
    print(f"  Target embeddings: {tgt_embedding:,}")
    print(f"  Positional encodings: {total_pos:,}")
    print(f"  Attention per layer: {attention_per_layer:,}")
    print(f"  Feedforward per layer: {ff_per_layer:,}")
    print(f"  Layer norm per layer: {ln_per_layer:,}")
    print(f"  Total per layer: {params_per_layer:,}")
    print(f"  All layers: {total_layers_params:,}")
    print(f"  Output projection: {output_projection:,}")
    print(f"  TOTAL: {total_params:,}")
    print()
    
    return total_params

def find_optimal_config_for_target(target_params=50000, vocab_size=200):
    """Find configuration that gets close to target parameter count"""
    
    print(f"🎯 Finding configuration for ~{target_params:,} parameters")
    print(f"   Fixed vocabulary size: {vocab_size}")
    print("="*60)
    
    best_config = None
    best_diff = float('inf')
    
    # Try different configurations
    configs_to_try = [
        # (d_model, seq_len, layers, heads, d_ff_multiplier)
        (32, 16, 1, 2, 1),
        (32, 16, 1, 2, 2), 
        (32, 32, 1, 2, 1),
        (48, 16, 1, 2, 1),
        (48, 16, 1, 3, 1),
        (64, 16, 1, 2, 1),
        (64, 16, 1, 4, 1),
        (64, 32, 1, 2, 1),
        (80, 16, 1, 2, 1),
        (96, 16, 1, 2, 1),
    ]
    
    for d_model, seq_len, layers, heads, d_ff_mult in configs_to_try:
        d_ff = d_model * d_ff_mult
        
        if d_model % heads != 0:  # d_model must be divisible by heads
            continue
            
        params = calculate_transformer_parameters(
            vocab_size, d_model, seq_len, layers, heads, d_ff
        )
        
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_config = (d_model, seq_len, layers, heads, d_ff, params)
        
        print("-" * 60)
    
    if best_config:
        d_model, seq_len, layers, heads, d_ff, params = best_config
        print(f"🏆 BEST CONFIGURATION:")
        print(f"   d_model: {d_model}")
        print(f"   seq_len: {seq_len}")
        print(f"   layers: {layers}")
        print(f"   heads: {heads}")
        print(f"   d_ff: {d_ff}")
        print(f"   Total parameters: {params:,}")
        print(f"   Difference from target: {abs(params - target_params):,}")
        
        return {
            "d_model": d_model,
            "seq_len": seq_len,
            "layers": layers, 
            "heads": heads,
            "d_ff": d_ff,
            "vocab_size": vocab_size
        }
    
    return None

if __name__ == "__main__":
    # Test current configuration
    print("Current configuration:")
    current_params = calculate_transformer_parameters(
        vocab_size=200,
        d_model=64, 
        seq_len=32,
        n_layers=1,
        n_heads=2,
        d_ff=128
    )
    
    print("="*60)
    
    # Find optimal for 50k parameters
    optimal = find_optimal_config_for_target(50000, 200)
    
    if optimal:
        print("\n" + "="*60)
        print("Recommended config.py settings:")
        print(f'"d_model": {optimal["d_model"]},')
        print(f'"seq_len": {optimal["seq_len"]},')
        print(f'"layers": {optimal["layers"]},')
        print(f'"heads": {optimal["heads"]},')
        print(f'"d_ff": {optimal["d_ff"]},')
        print(f'"vocab_size": {optimal["vocab_size"]},')