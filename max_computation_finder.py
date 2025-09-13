#!/usr/bin/env python3
"""
Find configuration that maximizes CPU computation time under 50k parameters
"""

def calculate_params(vocab_size, d_model, layers, heads, d_ff):
    """Calculate transformer parameters"""
    # Embeddings (src + tgt + output projection)
    embeddings = 3 * vocab_size * d_model
    
    # Per layer: attention (4*d_model²) + feedforward (2*d_model*d_ff) + layernorm (4*d_model)
    per_layer = 4 * d_model * d_model + 2 * d_model * d_ff + 4 * d_model
    
    # Total layers (encoder + decoder)
    layers_total = per_layer * layers * 2
    
    return embeddings + layers_total

def estimate_computation_complexity(vocab_size, d_model, seq_len, layers, heads, d_ff):
    """Estimate computational complexity (roughly proportional to inference time)"""
    
    # Attention computation: O(seq_len² * d_model * heads * layers)
    attention_ops = seq_len * seq_len * d_model * heads * layers * 2  # encoder + decoder
    
    # Feedforward computation: O(seq_len * d_model * d_ff * layers)  
    feedforward_ops = seq_len * d_model * d_ff * layers * 2  # encoder + decoder
    
    # Embeddings computation: O(seq_len * d_model)
    embedding_ops = seq_len * d_model * 2  # src + tgt
    
    # Output projection: O(seq_len * d_model * vocab_size)
    output_ops = seq_len * d_model * vocab_size
    
    total_ops = attention_ops + feedforward_ops + embedding_ops + output_ops
    return total_ops

def find_max_computation_config(max_params=50000):
    """Find config that maximizes computation time under parameter limit"""
    
    print(f"🎯 Finding configuration for MAXIMUM computation time under {max_params:,} parameters")
    print("="*80)
    
    candidates = []
    
    # Test configurations prioritizing computation over parameters
    for vocab_size in [100, 200, 400, 600, 800, 1000]:
        for d_model in [16, 24, 32, 48, 64, 80, 96]:
            for seq_len in [32, 64, 128, 256, 512]:  # Longer sequences = more computation
                for layers in [1, 2, 3, 4, 5]:  # More layers = more computation
                    for heads in [2, 4, 6, 8]:
                        if d_model % heads != 0:  # d_model must be divisible by heads
                            continue
                        for d_ff_mult in [1, 2, 3, 4, 6, 8]:
                            d_ff = d_model * d_ff_mult
                            
                            params = calculate_params(vocab_size, d_model, layers, heads, d_ff)
                            
                            if params <= max_params:
                                # Estimate computation complexity
                                computation = estimate_computation_complexity(
                                    vocab_size, d_model, seq_len, layers, heads, d_ff
                                )
                                
                                candidates.append({
                                    'vocab_size': vocab_size,
                                    'd_model': d_model,
                                    'seq_len': seq_len,
                                    'layers': layers,
                                    'heads': heads,
                                    'd_ff': d_ff,
                                    'params': params,
                                    'computation': computation,
                                    'comp_per_param': computation / params
                                })
    
    if not candidates:
        print("No valid configurations found!")
        return None
    
    # Sort by total computation (maximize inference time)
    candidates.sort(key=lambda x: x['computation'], reverse=True)
    
    print("Top configurations for MAXIMUM computation time:")
    print(f"{'Rank':<4} {'Params':<8} {'Vocab':<6} {'d_model':<8} {'seq_len':<8} {'Layers':<7} {'Heads':<6} {'d_ff':<5} {'Computation':<12}")
    print("-" * 80)
    
    for i, config in enumerate(candidates[:15]):
        comp_str = f"{config['computation']/1e9:.1f}B"
        print(f"{i+1:<4} {config['params']:<8,} {config['vocab_size']:<6} {config['d_model']:<8} "
              f"{config['seq_len']:<8} {config['layers']:<7} {config['heads']:<6} {config['d_ff']:<5} {comp_str:<12}")
    
    # Also show most computation-efficient (computation per parameter)
    print(f"\n" + "="*80)
    print("Top configurations for HIGHEST computation per parameter:")
    candidates_by_efficiency = sorted(candidates, key=lambda x: x['comp_per_param'], reverse=True)
    
    print(f"{'Rank':<4} {'Params':<8} {'Vocab':<6} {'d_model':<8} {'seq_len':<8} {'Layers':<7} {'Heads':<6} {'d_ff':<5} {'Comp/Param':<12}")
    print("-" * 80)
    
    for i, config in enumerate(candidates_by_efficiency[:10]):
        efficiency = config['comp_per_param'] / 1e6
        print(f"{i+1:<4} {config['params']:<8,} {config['vocab_size']:<6} {config['d_model']:<8} "
              f"{config['seq_len']:<8} {config['layers']:<7} {config['heads']:<6} {config['d_ff']:<5} {efficiency:<12.1f}M")
    
    # Recommend best options
    max_computation = candidates[0]
    max_efficiency = candidates_by_efficiency[0]
    
    print(f"\n🏆 MAXIMUM COMPUTATION TIME CONFIG:")
    print(f"   vocab_size: {max_computation['vocab_size']}")
    print(f"   d_model: {max_computation['d_model']}")
    print(f"   seq_len: {max_computation['seq_len']}")
    print(f"   layers: {max_computation['layers']}")
    print(f"   heads: {max_computation['heads']}")
    print(f"   d_ff: {max_computation['d_ff']}")
    print(f"   Parameters: {max_computation['params']:,}")
    print(f"   Estimated computation: {max_computation['computation']/1e9:.1f}B operations")
    
    print(f"\n⚡ MOST COMPUTATION-EFFICIENT CONFIG:")
    print(f"   vocab_size: {max_efficiency['vocab_size']}")
    print(f"   d_model: {max_efficiency['d_model']}")
    print(f"   seq_len: {max_efficiency['seq_len']}")
    print(f"   layers: {max_efficiency['layers']}")
    print(f"   heads: {max_efficiency['heads']}")
    print(f"   d_ff: {max_efficiency['d_ff']}")
    print(f"   Parameters: {max_efficiency['params']:,}")
    print(f"   Computation per parameter: {max_efficiency['comp_per_param']/1e6:.1f}M ops/param")
    
    # Estimate actual inference times
    print(f"\n⏱️  ESTIMATED INFERENCE TIMES:")
    print(f"   Current config (seq_len=64): ~27ms")
    
    # Scale based on computation complexity relative to current
    current_computation = estimate_computation_complexity(200, 32, 64, 1, 8, 96)  # Current config
    max_time_ratio = max_computation['computation'] / current_computation
    efficiency_time_ratio = max_efficiency['computation'] / current_computation
    
    print(f"   Maximum computation: ~{27 * max_time_ratio:.0f}ms ({max_time_ratio:.1f}x longer)")
    print(f"   Most efficient: ~{27 * efficiency_time_ratio:.0f}ms ({efficiency_time_ratio:.1f}x longer)")
    
    return max_computation, max_efficiency

if __name__ == "__main__":
    find_max_computation_config()