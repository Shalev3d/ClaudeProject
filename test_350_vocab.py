#!/usr/bin/env python3
"""
Test configurations with 300-350 vocabulary size for longer inference times
"""

import torch
import time
from model import build_transformer
from force_long_inference import create_long_sequences

def test_vocab_size_configurations():
    """Test different vocabulary sizes around 300-350"""
    
    print("🎯 TESTING 300-350 VOCABULARY CONFIGURATIONS")
    print("="*60)
    
    # Test different vocabulary sizes
    vocab_configs = [
        # (vocab_size, d_model, seq_len, layers, heads, d_ff)
        (300, 20, 1024, 4, 4, 20),
        (350, 18, 1024, 4, 2, 18), 
        (320, 20, 1024, 3, 4, 20),
        (300, 24, 512, 5, 4, 24),
        (350, 20, 512, 5, 4, 20),
    ]
    
    results = []
    
    for vocab_size, d_model, seq_len, layers, heads, d_ff in vocab_configs:
        try:
            print(f"\n🧪 Testing vocab_size={vocab_size}, d_model={d_model}, seq_len={seq_len}, layers={layers}")
            
            # Build model
            model = build_transformer(
                src_vocab_size=vocab_size,
                tgt_vocab_size=vocab_size,
                src_seq_len=seq_len,
                tgt_seq_len=seq_len,
                d_model=d_model,
                N=layers,
                h=heads,
                dropout=0.1,
                d_ff=d_ff
            )
            
            params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {params:,}")
            
            if params > 50000:
                print(f"   ❌ Over 50k parameter limit")
                continue
            
            # Create test sequences
            encoder_input, decoder_input, encoder_mask, decoder_mask = create_long_sequences(
                vocab_size, seq_len
            )
            
            # Benchmark inference
            model.eval()
            times = []
            
            for _ in range(3):  # 3 runs for average
                start_time = time.time()
                with torch.no_grad():
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    projection_output = model.project(decoder_output)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            
            # Calculate embedding vs architecture parameter allocation
            embeddings = 3 * vocab_size * d_model
            architecture = params - embeddings
            emb_pct = (embeddings / params) * 100
            arch_pct = (architecture / params) * 100
            
            result = {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'seq_len': seq_len,
                'layers': layers,
                'params': params,
                'inference_time_ms': avg_time * 1000,
                'inference_time_s': avg_time,
                'embeddings_pct': emb_pct,
                'architecture_pct': arch_pct
            }
            results.append(result)
            
            print(f"   ✅ Inference time: {avg_time*1000:.1f} ms ({avg_time:.3f} seconds)")
            print(f"   Parameter allocation: {emb_pct:.1f}% embeddings, {arch_pct:.1f}% architecture")
            
            if avg_time > 1.0:
                print(f"   🎉 SUCCESS! Over 1 second!")
                
        except Exception as e:
            print(f"   ❌ Config failed: {e}")
    
    # Show results summary
    if results:
        print(f"\n📊 RESULTS SUMMARY")
        print("="*80)
        print(f"{'Vocab':<6} {'d_model':<8} {'seq_len':<8} {'Layers':<7} {'Params':<8} {'Time(ms)':<9} {'Time(s)':<8} {'Emb%':<5} {'Arch%':<6}")
        print("-" * 80)
        
        # Sort by inference time (longest first)
        results.sort(key=lambda x: x['inference_time_s'], reverse=True)
        
        for r in results:
            print(f"{r['vocab_size']:<6} {r['d_model']:<8} {r['seq_len']:<8} {r['layers']:<7} "
                  f"{r['params']:<8,} {r['inference_time_ms']:<9.1f} {r['inference_time_s']:<8.3f} "
                  f"{r['embeddings_pct']:<5.1f} {r['architecture_pct']:<6.1f}")
        
        best = results[0]
        print(f"\n🏆 BEST CONFIGURATION (longest inference time):")
        print(f"   vocab_size: {best['vocab_size']}")
        print(f"   d_model: {best['d_model']}")
        print(f"   seq_len: {best['seq_len']}")
        print(f"   layers: {best['layers']}")
        print(f"   Parameters: {best['params']:,}")
        print(f"   Inference time: {best['inference_time_ms']:.1f} ms ({best['inference_time_s']:.3f} seconds)")
        
        return best
    
    return None

def create_config_for_vocab_size(vocab_size):
    """Create optimized configuration for specific vocabulary size"""
    
    print(f"\n🔧 OPTIMIZING CONFIGURATION FOR vocab_size={vocab_size}")
    print("="*55)
    
    best_config = None
    best_time = 0
    
    # Try different combinations within parameter budget
    for d_model in [16, 18, 20, 24]:
        for seq_len in [512, 768, 1024]:
            for layers in [3, 4, 5, 6]:
                for heads in [2, 4]:
                    if d_model % heads != 0:
                        continue
                    
                    d_ff = d_model  # Keep d_ff = d_model for efficiency
                    
                    try:
                        model = build_transformer(
                            src_vocab_size=vocab_size,
                            tgt_vocab_size=vocab_size,
                            src_seq_len=seq_len,
                            tgt_seq_len=seq_len,
                            d_model=d_model,
                            N=layers,
                            h=heads,
                            dropout=0.1,
                            d_ff=d_ff
                        )
                        
                        params = sum(p.numel() for p in model.parameters())
                        
                        if params <= 50000:
                            # Quick timing test
                            encoder_input, decoder_input, encoder_mask, decoder_mask = create_long_sequences(
                                vocab_size, seq_len
                            )
                            
                            model.eval()
                            start_time = time.time()
                            with torch.no_grad():
                                encoder_output = model.encode(encoder_input, encoder_mask)
                                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                                projection_output = model.project(decoder_output)
                            end_time = time.time()
                            
                            inference_time = end_time - start_time
                            
                            if inference_time > best_time:
                                best_time = inference_time
                                best_config = {
                                    'vocab_size': vocab_size,
                                    'd_model': d_model,
                                    'seq_len': seq_len,
                                    'layers': layers,
                                    'heads': heads,
                                    'd_ff': d_ff,
                                    'params': params,
                                    'inference_time': inference_time
                                }
                                
                    except Exception:
                        continue
    
    if best_config:
        print(f"OPTIMAL CONFIG for vocab_size={vocab_size}:")
        for key, value in best_config.items():
            if key == 'inference_time':
                print(f"   {key}: {value*1000:.1f} ms ({value:.3f} seconds)")
            elif key == 'params':
                print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value}")
                
        return best_config
    
    return None

if __name__ == "__main__":
    # Test various 300-350 vocabulary configurations
    best = test_vocab_size_configurations()
    
    # Try to optimize specifically for 350 vocab
    optimal_350 = create_config_for_vocab_size(350)
    
    # Try to optimize specifically for 300 vocab  
    optimal_300 = create_config_for_vocab_size(300)