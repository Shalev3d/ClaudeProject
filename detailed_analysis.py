#!/usr/bin/env python3
"""
Detailed analysis of how each configuration parameter affects the total parameter count
"""

def analyze_parameter_breakdown(vocab_size, d_model, seq_len, n_layers, n_heads, d_ff):
    """
    Detailed breakdown of transformer parameters by component
    """
    print(f"Configuration: vocab={vocab_size}, d_model={d_model}, seq_len={seq_len}, layers={n_layers}, heads={n_heads}, d_ff={d_ff}")
    print("=" * 100)
    
    # 1. EMBEDDING LAYERS
    print("1. EMBEDDING LAYERS:")
    src_embedding = vocab_size * d_model
    tgt_embedding = vocab_size * d_model
    total_embeddings = src_embedding + tgt_embedding
    print(f"   Source embeddings: {vocab_size} × {d_model} = {src_embedding:,}")
    print(f"   Target embeddings: {vocab_size} × {d_model} = {tgt_embedding:,}")
    print(f"   Total embeddings: {total_embeddings:,}")
    print(f"   → Linear scaling with vocabulary size and d_model")
    
    # 2. POSITIONAL ENCODINGS (usually fixed, not learned)
    print(f"\n2. POSITIONAL ENCODINGS (usually not learned):")
    src_pos = seq_len * d_model
    tgt_pos = seq_len * d_model
    total_pos = src_pos + tgt_pos
    print(f"   Source positional: {seq_len} × {d_model} = {src_pos:,}")
    print(f"   Target positional: {seq_len} × {d_model} = {tgt_pos:,}")
    print(f"   Total positional: {total_pos:,}")
    print(f"   → Linear scaling with sequence length and d_model")
    
    # 3. MULTI-HEAD ATTENTION (per layer)
    print(f"\n3. MULTI-HEAD ATTENTION (per layer):")
    q_params = d_model * d_model
    k_params = d_model * d_model  
    v_params = d_model * d_model
    o_params = d_model * d_model
    attention_per_layer = q_params + k_params + v_params + o_params
    print(f"   Query projection (W_Q): {d_model} × {d_model} = {q_params:,}")
    print(f"   Key projection (W_K): {d_model} × {d_model} = {k_params:,}")
    print(f"   Value projection (W_V): {d_model} × {d_model} = {v_params:,}")
    print(f"   Output projection (W_O): {d_model} × {d_model} = {o_params:,}")
    print(f"   Total attention per layer: {attention_per_layer:,}")
    print(f"   → QUADRATIC scaling with d_model (4 × d_model²)")
    print(f"   → Number of heads doesn't affect param count (just reshapes existing params)")
    
    # 4. FEED-FORWARD NETWORK (per layer)
    print(f"\n4. FEED-FORWARD NETWORK (per layer):")
    ff1_params = d_model * d_ff
    ff2_params = d_ff * d_model
    ff_per_layer = ff1_params + ff2_params
    print(f"   Linear 1: {d_model} × {d_ff} = {ff1_params:,}")
    print(f"   Linear 2: {d_ff} × {d_model} = {ff2_params:,}")
    print(f"   Total FF per layer: {ff_per_layer:,}")
    print(f"   → Linear scaling with both d_model and d_ff")
    
    # 5. LAYER NORMALIZATION (per layer)
    print(f"\n5. LAYER NORMALIZATION (per layer):")
    ln_after_attn = 2 * d_model  # gamma and beta
    ln_after_ff = 2 * d_model    # gamma and beta
    ln_per_layer = ln_after_attn + ln_after_ff
    print(f"   After attention: 2 × {d_model} = {ln_after_attn:,}")
    print(f"   After feed-forward: 2 × {d_model} = {ln_after_ff:,}")
    print(f"   Total LN per layer: {ln_per_layer:,}")
    print(f"   → Linear scaling with d_model")
    
    # 6. LAYER TOTALS
    print(f"\n6. LAYER CALCULATIONS:")
    params_per_layer = attention_per_layer + ff_per_layer + ln_per_layer
    # Each transformer layer appears in both encoder AND decoder
    encoder_layers = params_per_layer * n_layers
    decoder_layers = params_per_layer * n_layers  # Decoder has extra cross-attention but similar count
    total_layers_params = encoder_layers + decoder_layers
    print(f"   Parameters per layer: {params_per_layer:,}")
    print(f"   Encoder layers: {n_layers} × {params_per_layer:,} = {encoder_layers:,}")
    print(f"   Decoder layers: {n_layers} × {params_per_layer:,} = {decoder_layers:,}")
    print(f"   Total layer params: {total_layers_params:,}")
    print(f"   → Linear scaling with number of layers")
    
    # 7. OUTPUT PROJECTION
    print(f"\n7. OUTPUT PROJECTION:")
    output_projection = d_model * vocab_size
    print(f"   Final projection: {d_model} × {vocab_size} = {output_projection:,}")
    print(f"   → Linear scaling with d_model and vocabulary size")
    
    # 8. TOTAL
    print(f"\n8. TOTAL PARAMETERS:")
    total_params = total_embeddings + total_pos + total_layers_params + output_projection
    print(f"   Embeddings: {total_embeddings:,}")
    print(f"   Positional: {total_pos:,}")
    print(f"   Layers: {total_layers_params:,}")
    print(f"   Output proj: {output_projection:,}")
    print(f"   GRAND TOTAL: {total_params:,}")
    
    # 9. SCALING ANALYSIS
    print(f"\n9. PARAMETER SCALING BEHAVIOR:")
    embedding_percent = (total_embeddings / total_params) * 100
    layers_percent = (total_layers_params / total_params) * 100
    output_percent = (output_projection / total_params) * 100
    
    print(f"   Embeddings: {embedding_percent:.1f}% of total")
    print(f"   Layers: {layers_percent:.1f}% of total") 
    print(f"   Output projection: {output_percent:.1f}% of total")
    
    if embedding_percent > 50:
        print(f"   ⚠️  VOCABULARY-DOMINATED: Embeddings consume most parameters")
    elif layers_percent > 50:
        print(f"   🏗️  ARCHITECTURE-DOMINATED: Layer parameters dominate")
    
    return total_params

def compare_configurations():
    """Compare different configurations to show scaling effects"""
    
    print("\n" + "="*100)
    print("CONFIGURATION COMPARISON - SCALING EFFECTS")
    print("="*100)
    
    configs = [
        ("Tiny Vocab", 200, 32, 16, 1, 2, 32),
        ("Medium Vocab", 2000, 32, 16, 1, 2, 32),
        ("Large Vocab", 20000, 32, 16, 1, 2, 32),
        ("Huge Vocab", 100000, 32, 16, 1, 2, 32),
        
        ("Small d_model", 2000, 32, 16, 1, 2, 32),
        ("Medium d_model", 2000, 64, 16, 1, 4, 64),
        ("Large d_model", 2000, 128, 16, 1, 8, 128),
        
        ("1 Layer", 2000, 64, 16, 1, 4, 64),
        ("2 Layers", 2000, 64, 16, 2, 4, 64),
        ("4 Layers", 2000, 64, 16, 4, 4, 64),
        ("8 Layers", 2000, 64, 16, 8, 4, 64),
    ]
    
    results = []
    for name, vocab, d_model, seq_len, layers, heads, d_ff in configs:
        params = analyze_parameter_breakdown(vocab, d_model, seq_len, layers, heads, d_ff)
        results.append((name, params, vocab, d_model, layers, d_ff))
        print("\n" + "-"*100 + "\n")
    
    # Show scaling effects
    print("="*100)
    print("SCALING EFFECT SUMMARY")
    print("="*100)
    
    # Vocabulary scaling
    vocab_configs = [r for r in results if "Vocab" in r[0]]
    if len(vocab_configs) > 1:
        print(f"\n📊 VOCABULARY SIZE SCALING:")
        base = vocab_configs[0]
        for name, params, vocab, d_model, layers, d_ff in vocab_configs:
            ratio = params / base[1]
            vocab_ratio = vocab / base[2]
            print(f"   {name}: {params:,} params ({ratio:.1f}x, vocab {vocab_ratio:.0f}x)")
    
    # d_model scaling  
    d_model_configs = [r for r in results if "d_model" in r[0]]
    if len(d_model_configs) > 1:
        print(f"\n📊 MODEL DIMENSION SCALING:")
        base = d_model_configs[0]
        for name, params, vocab, d_model, layers, d_ff in d_model_configs:
            ratio = params / base[1]
            d_ratio = d_model / base[3]
            print(f"   {name}: {params:,} params ({ratio:.1f}x, d_model {d_ratio:.1f}x)")
            
    # Layer scaling
    layer_configs = [r for r in results if "Layer" in r[0]]
    if len(layer_configs) > 1:
        print(f"\n📊 LAYER COUNT SCALING:")
        base = layer_configs[0]
        for name, params, vocab, d_model, layers, d_ff in layer_configs:
            ratio = params / base[1]
            layer_ratio = layers / base[4]
            print(f"   {name}: {params:,} params ({ratio:.1f}x, {layer_ratio:.0f}x layers)")

if __name__ == "__main__":
    compare_configurations()