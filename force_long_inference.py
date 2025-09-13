#!/usr/bin/env python3
"""
Force truly long inference times by creating artificial long sequences
"""

import torch
import time
from model import build_transformer
from config import get_config

def create_long_sequences(vocab_size, seq_len, batch_size=1):
    """Create artificial sequences that use the full seq_len"""
    
    # Create random token sequences of exactly seq_len length
    encoder_input = torch.randint(1, vocab_size-1, (batch_size, seq_len))
    decoder_input = torch.randint(1, vocab_size-1, (batch_size, seq_len))
    
    # Create attention masks (all ones = attend to all positions)
    encoder_mask = torch.ones((batch_size, 1, 1, seq_len))
    decoder_mask = torch.ones((batch_size, 1, seq_len, seq_len))
    
    # Make decoder mask causal (lower triangular)
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            decoder_mask[:, :, i, j] = 0
    
    return encoder_input, decoder_input, encoder_mask, decoder_mask

def benchmark_full_length_inference():
    """Benchmark inference with FULL 512-token sequences"""
    
    config = get_config()
    print(f"🚀 FORCING FULL {config['seq_len']}-TOKEN INFERENCE")
    print("="*60)
    
    # Build model
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
    print(f"Model: {params:,} parameters")
    print(f"Architecture: {config['layers']} layers, {config['seq_len']} seq_len")
    print()
    
    # Create artificial full-length sequences
    encoder_input, decoder_input, encoder_mask, decoder_mask = create_long_sequences(
        config['vocab_size'], config['seq_len']
    )
    
    print(f"Testing with ACTUAL {config['seq_len']}-token sequences:")
    print(f"  • Encoder input shape: {encoder_input.shape}")
    print(f"  • Decoder input shape: {decoder_input.shape}")
    print(f"  • Every token position will be processed")
    print()
    
    model.eval()
    device = "cpu"
    
    # Warmup
    with torch.no_grad():
        _ = model.encode(encoder_input, encoder_mask)
    
    # Benchmark multiple runs
    times = []
    num_runs = 5
    
    print(f"Running {num_runs} inference tests...")
    
    for i in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            # Full forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            projection_output = model.project(decoder_output)
        
        end_time = time.time()
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"  Run {i+1}: {inference_time*1000:.1f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 RESULTS with FULL {config['seq_len']}-token sequences:")
    print(f"  • Average inference time: {avg_time*1000:.1f} ms ({avg_time:.3f} seconds)")
    print(f"  • Tokens processed: {config['seq_len']}")
    print(f"  • Layers processed: {config['layers']} encoder + {config['layers']} decoder = {config['layers']*2} total")
    print(f"  • Total operations: {config['seq_len']**2 * config['layers'] * 2:,} (quadratic attention)")
    
    # Calculate cycles estimate
    cpu_freq = 3.0e9  # 3 GHz
    estimated_cycles = avg_time * cpu_freq
    
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    print(f"  • Estimated CPU cycles: {estimated_cycles:,.0f}")
    print(f"  • Cycles per token: {estimated_cycles/config['seq_len']:,.0f}")
    print(f"  • Tokens per second: {config['seq_len']/avg_time:.0f}")
    
    # FPGA analysis
    fpga_freq = 100e6  # 100 MHz
    fpga_cycles = estimated_cycles * (cpu_freq / fpga_freq)
    fpga_time = fpga_cycles / fpga_freq
    
    print(f"\n🔧 FPGA ESTIMATES:")
    print(f"  • FPGA cycles (100MHz): {fpga_cycles:,.0f}")
    print(f"  • FPGA time estimate: {fpga_time*1000:.1f} ms")
    print(f"  • FPGA speedup potential: {avg_time/fpga_time:.2f}x")
    
    if avg_time/fpga_time > 1.0:
        print(f"  🎉 FPGA would be FASTER than CPU!")
    else:
        print(f"  😞 FPGA still slower due to overhead")
    
    return avg_time

def test_different_sequence_lengths():
    """Test inference time scaling with sequence length"""
    
    config = get_config()
    print(f"\n🔬 SEQUENCE LENGTH SCALING TEST")
    print("="*50)
    
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
    
    model.eval()
    seq_lengths = [64, 128, 256, 512]
    
    print(f"Testing different sequence lengths (max {config['seq_len']}):")
    
    for seq_len in seq_lengths:
        if seq_len > config['seq_len']:
            continue
            
        encoder_input, decoder_input, encoder_mask, decoder_mask = create_long_sequences(
            config['vocab_size'], seq_len
        )
        
        # Trim masks to match sequence length
        encoder_mask = encoder_mask[:, :, :, :seq_len]
        decoder_mask = decoder_mask[:, :, :seq_len, :seq_len]
        
        start_time = time.time()
        with torch.no_grad():
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            projection_output = model.project(decoder_output)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000
        tokens_per_sec = seq_len / ((end_time - start_time))
        
        print(f"  seq_len {seq_len:>3}: {inference_time:>6.1f} ms ({tokens_per_sec:>5.0f} tokens/sec)")

if __name__ == "__main__":
    # Test with full-length sequences
    avg_time = benchmark_full_length_inference()
    
    # Test scaling with sequence length
    test_different_sequence_lengths()
    
    print(f"\n🎯 CONCLUSION:")
    if avg_time > 1.0:  # More than 1 second
        print(f"   SUCCESS! Achieved {avg_time:.1f} second inference time")
    else:
        print(f"   Still only {avg_time*1000:.1f} ms - need even longer sequences or more layers")
        print("   Consider: seq_len=1024, or more layers within 50k parameter limit")