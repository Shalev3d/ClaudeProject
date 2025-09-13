#!/usr/bin/env python3
"""
Find optimal FPGA-friendly configuration under parameter constraint
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

def find_fpga_optimal_config(max_params=48000):
    """Find FPGA-friendly config under parameter constraint"""
    
    print(f"🎯 Finding FPGA-optimal config under {max_params:,} parameters")
    print("="*70)
    
    candidates = []
    
    # FPGA prefers: larger d_model (parallel matrix ops), smaller vocab (less memory)
    # Test configurations prioritizing larger d_model
    for vocab_size in [200, 400, 600, 800]:
        for d_model in [32, 48, 64, 80, 96]:  # Must be divisible by heads
            for layers in [1, 2]:
                for heads in [2, 4, 8]:
                    if d_model % heads != 0:  # d_model must be divisible by heads
                        continue
                    for d_ff_mult in [1, 2, 4]:  # d_ff as multiple of d_model
                        d_ff = d_model * d_ff_mult
                        
                        params = calculate_params(vocab_size, d_model, layers, heads, d_ff)
                        
                        if params <= max_params:
                            # FPGA efficiency score: prioritize larger d_model, more parallel ops
                            fpga_score = (
                                d_model * 2 +           # Larger matrices = better parallelization
                                heads * 10 +            # More attention heads = more parallel
                                layers * 5 +            # More computation per inference
                                (d_ff / d_model) * 3    # Larger feedforward = more compute
                            ) / (vocab_size / 1000)     # Penalize large vocab (memory intensive)
                            
                            candidates.append({
                                'vocab_size': vocab_size,
                                'd_model': d_model,
                                'layers': layers,
                                'heads': heads,
                                'd_ff': d_ff,
                                'params': params,
                                'fpga_score': fpga_score
                            })
    
    # Sort by FPGA efficiency score
    candidates.sort(key=lambda x: x['fpga_score'], reverse=True)
    
    print("Top FPGA-Optimized Configurations:")
    print(f"{'Rank':<4} {'Params':<8} {'Vocab':<6} {'d_model':<8} {'Layers':<7} {'Heads':<6} {'d_ff':<5} {'FPGA Score':<10}")
    print("-" * 70)
    
    for i, config in enumerate(candidates[:10]):
        print(f"{i+1:<4} {config['params']:<8,} {config['vocab_size']:<6} {config['d_model']:<8} "
              f"{config['layers']:<7} {config['heads']:<6} {config['d_ff']:<5} {config['fpga_score']:<10.1f}")
    
    if candidates:
        best = candidates[0]
        print(f"\n🏆 RECOMMENDED FPGA-OPTIMAL CONFIG:")
        print(f"   vocab_size: {best['vocab_size']}")
        print(f"   d_model: {best['d_model']}")
        print(f"   layers: {best['layers']}")
        print(f"   heads: {best['heads']}")  
        print(f"   d_ff: {best['d_ff']}")
        print(f"   Total parameters: {best['params']:,}")
        print(f"   FPGA efficiency score: {best['fpga_score']:.1f}")
        
        print(f"\n📋 Why this is FPGA-friendly:")
        print(f"   • Large d_model ({best['d_model']}) = better matrix parallelization")
        print(f"   • Multiple heads ({best['heads']}) = parallel attention computation")
        print(f"   • Moderate vocab ({best['vocab_size']}) = efficient memory usage")
        print(f"   • d_ff ratio ({best['d_ff']//best['d_model']}x) = substantial computation per layer")
        
        return best
    
    return None

def compare_current_vs_optimal():
    """Compare current config vs FPGA-optimal"""
    from config import get_config
    current = get_config()
    
    print(f"\n{'='*70}")
    print("CURRENT vs FPGA-OPTIMAL COMPARISON")
    print(f"{'='*70}")
    
    current_params = calculate_params(
        current['vocab_size'], current['d_model'], 
        current['layers'], current['heads'], current['d_ff']
    )
    
    print(f"CURRENT CONFIG:")
    print(f"   vocab_size: {current['vocab_size']}")
    print(f"   d_model: {current['d_model']}")
    print(f"   layers: {current['layers']}")
    print(f"   heads: {current['heads']}")
    print(f"   d_ff: {current['d_ff']}")
    print(f"   Parameters: {current_params:,}")
    
    optimal = find_fpga_optimal_config()
    
    if optimal:
        print(f"\nPARAMETER ALLOCATION COMPARISON:")
        
        # Current allocation
        curr_emb = 3 * current['vocab_size'] * current['d_model']
        curr_arch = current_params - curr_emb
        curr_emb_pct = (curr_emb / current_params) * 100
        curr_arch_pct = (curr_arch / current_params) * 100
        
        # Optimal allocation  
        opt_emb = 3 * optimal['vocab_size'] * optimal['d_model']
        opt_arch = optimal['params'] - opt_emb
        opt_emb_pct = (opt_emb / optimal['params']) * 100
        opt_arch_pct = (opt_arch / optimal['params']) * 100
        
        print(f"                    Current     Optimal")
        print(f"   Embeddings:      {curr_emb_pct:5.1f}%      {opt_emb_pct:5.1f}%")
        print(f"   Architecture:    {curr_arch_pct:5.1f}%      {opt_arch_pct:5.1f}%")
        print(f"\n💡 FPGA Insight: More architecture % = better parallelization!")

if __name__ == "__main__":
    compare_current_vs_optimal()