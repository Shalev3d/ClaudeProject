#!/usr/bin/env python3
"""
Comprehensive transformer benchmarking with cycle counting
"""

import torch
import sys
from config import get_config
from train import get_model
from cycle_counter import count_transformer_cycles, detailed_layer_profiling, benchmark_model_sizes, cycle_counter
import time
import numpy as np


def benchmark_current_model():
    """Benchmark the current model configuration"""
    print("🏁 Benchmarking Current Transformer Model")
    print("=" * 60)
    
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create dummy tokenizer sizes (since we just need vocab size)
    vocab_size = config['vocab_size'] if config['vocab_size'] else 200
    
    # Build model
    model = get_model(config, vocab_size, vocab_size)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture:")
    print(f"  • Parameters: {total_params:,}")
    print(f"  • Trainable: {trainable_params:,}")
    print(f"  • Vocabulary: {vocab_size}")
    print(f"  • Sequence length: {config['seq_len']}")
    print(f"  • Model dimension: {config['d_model']}")
    print(f"  • Layers: {config['layers']}")
    print(f"  • Heads: {config['heads']}")
    print()
    
    # Create test inputs
    batch_size = 1
    seq_len = config['seq_len']
    
    encoder_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoder_mask = torch.ones(batch_size, 1, 1, seq_len)
    decoder_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Create causal mask for decoder
    decoder_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    # Warmup runs
    print("🔥 Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model.encode(encoder_input.to(device), encoder_mask.to(device))
    
    # Benchmark multiple runs
    print("⏱️  Running benchmark...")
    num_runs = 10
    times = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            encoder_output = model.encode(encoder_input.to(device), encoder_mask.to(device))
            decoder_output = model.decode(encoder_output, encoder_mask.to(device), 
                                        decoder_input.to(device), decoder_mask.to(device))
            output = model.project(decoder_output)
        
        if device == "cuda":
            torch.cuda.synchronize()
            
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        
        if i == 0:  # Detailed timing for first run
            print(f"Run {i+1}: {elapsed*1000:.3f} ms")
    
    # Statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\n📊 Timing Statistics ({num_runs} runs):")
    print(f"  • Average: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")
    print(f"  • Min: {min_time*1000:.3f} ms")
    print(f"  • Max: {max_time*1000:.3f} ms")
    print(f"  • Tokens/second: {seq_len/avg_time:.0f}")
    
    # Detailed cycle analysis
    print("\n🔍 Detailed Cycle Analysis:")
    stats = count_transformer_cycles(model, encoder_input, encoder_mask, 
                                   decoder_input, decoder_mask, device)
    cycle_counter.print_summary()
    
    # Layer-by-layer profiling
    print("\n🏗️  Layer Profiling:")
    detailed_stats = detailed_layer_profiling(model, encoder_input, encoder_mask,
                                            decoder_input, decoder_mask, device)
    
    # Print layer breakdown
    print(f"\nLayer Breakdown:")
    for operation, data in sorted(detailed_stats.items(), key=lambda x: x[1]['avg_time'], reverse=True):
        if 'layer' in operation or 'embedding' in operation or 'projection' in operation:
            print(f"  • {operation}: {data['avg_time']*1000:.3f} ms ({data['avg_cycles']:,.0f} cycles)")
    
    # FPGA projections
    print(f"\n🔧 FPGA Performance Projections:")
    cpu_freq = cycle_counter._cpu_freq
    total_cycles = sum(data['total_cycles'] for data in stats.values())
    
    fpga_freqs = [50e6, 100e6, 200e6, 500e6]  # Different FPGA clock frequencies
    
    for fpga_freq in fpga_freqs:
        fpga_cycles_est = total_cycles * (cpu_freq / fpga_freq)
        fpga_time_est = fpga_cycles_est / fpga_freq
        speedup = avg_time / fpga_time_est
        
        print(f"  • @ {fpga_freq/1e6:.0f} MHz: {fpga_time_est*1000:.3f} ms "
              f"({fpga_cycles_est:,.0f} cycles, {speedup:.1f}x speedup)")
    
    # Memory analysis
    if device == "cuda":
        print(f"\n💾 GPU Memory:")
        print(f"  • Allocated: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")
        print(f"  • Cached: {torch.cuda.memory_reserved()/1024/1024:.1f} MB")
    
    return {
        'avg_time': avg_time,
        'total_cycles': total_cycles,
        'parameters': total_params,
        'tokens_per_second': seq_len / avg_time
    }


def compare_model_sizes():
    """Compare different model configurations"""
    print("\n🔬 Model Size Comparison")
    print("=" * 60)
    
    configurations = [
        # (vocab, d_model, seq_len, name)
        (200, 32, 16, "Tiny"),
        (200, 48, 16, "Ultra-Small (Current)"), 
        (200, 64, 16, "Small"),
        (200, 48, 32, "Current + Longer Seq"),
        (500, 48, 16, "Current + Larger Vocab"),
    ]
    
    results = []
    
    for vocab_size, d_model, seq_len, name in configurations:
        try:
            print(f"\n🧪 Testing {name}: vocab={vocab_size}, d_model={d_model}, seq_len={seq_len}")
            
            # Create temporary config
            temp_config = {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'seq_len': seq_len,
                'layers': 1,
                'heads': 2,
                'd_ff': d_model
            }
            
            # Build model
            from model import build_transformer
            model = build_transformer(
                vocab_size, vocab_size, seq_len, seq_len,
                d_model=d_model, N=1, h=2, d_ff=d_model
            )
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            
            # Quick benchmark
            device = "cpu"  # Use CPU for comparison consistency
            model = model.to(device)
            model.eval()
            
            enc_input = torch.randint(0, vocab_size, (1, seq_len))
            enc_mask = torch.ones(1, 1, 1, seq_len)
            dec_input = torch.randint(0, vocab_size, (1, seq_len))
            dec_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            
            # Time single inference
            start_time = time.perf_counter()
            with torch.no_grad():
                encoder_output = model.encode(enc_input, enc_mask)
                decoder_output = model.decode(encoder_output, enc_mask, dec_input, dec_mask)
                output = model.project(decoder_output)
            elapsed = time.perf_counter() - start_time
            
            # Estimate cycles
            cpu_freq = 3e9  # 3 GHz
            cycles = int(elapsed * cpu_freq)
            
            results.append({
                'name': name,
                'parameters': params,
                'time_ms': elapsed * 1000,
                'cycles': cycles,
                'memory_mb': params * 4 / 1024 / 1024  # Assume 4 bytes per parameter
            })
            
            print(f"   Parameters: {params:,}")
            print(f"   Time: {elapsed*1000:.3f} ms")
            print(f"   Cycles: {cycles:,.0f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary table
    print(f"\n📋 Comparison Summary:")
    print(f"{'Model':<20} {'Parameters':<12} {'Time (ms)':<12} {'Cycles':<15} {'Memory (MB)':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<20} "
              f"{result['parameters']:<12,} "
              f"{result['time_ms']:<12.3f} "
              f"{result['cycles']:<15,.0f} "
              f"{result['memory_mb']:<12.1f}")
    
    return results


if __name__ == "__main__":
    print("🚀 Transformer Performance Benchmark")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_model_sizes()
    else:
        benchmark_current_model()
    
    print(f"\n✅ Benchmark complete!")
    print(f"💡 Use --compare flag to compare different model sizes")