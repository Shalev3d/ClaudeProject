#!/usr/bin/env python3
"""
Benchmark different transformer configurations for speed and memory usage
"""

import torch
import time
from model import build_transformer

def get_memory_usage():
    """Get current memory usage estimate"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0  # Simplified for CPU

def benchmark_config(config_name, config, num_runs=10):
    """Benchmark a specific configuration"""
    print(f"\nBenchmarking: {config_name}")
    print("-" * 50)
    
    # Build transformer
    try:
        transformer = build_transformer(
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
        
        # Count parameters
        param_count = sum(p.numel() for p in transformer.parameters())
        print(f"Parameters: {param_count:,}")
        
        # Create sample input
        batch_size = 4
        src_input = torch.randint(0, config['vocab_size'], (batch_size, config['seq_len']))
        tgt_input = torch.randint(0, config['vocab_size'], (batch_size, config['seq_len']))
        
        # Create masks (all ones for simplicity)
        src_mask = torch.ones((batch_size, 1, 1, config['seq_len']))
        tgt_mask = torch.ones((batch_size, 1, config['seq_len'], config['seq_len']))
        
        # Warm up
        transformer.eval()
        with torch.no_grad():
            encoder_output = transformer.encode(src_input, src_mask)
            decoder_output = transformer.decode(encoder_output, src_mask, tgt_input, tgt_mask)
            output = transformer.project(decoder_output)
        
        # Benchmark forward pass
        memory_before = get_memory_usage()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                encoder_output = transformer.encode(src_input, src_mask)
                decoder_output = transformer.decode(encoder_output, src_mask, tgt_input, tgt_mask)
                output = transformer.project(decoder_output)
            end_time = time.time()
            times.append(end_time - start_time)
        
        memory_after = get_memory_usage()
        
        avg_time = sum(times) / len(times)
        memory_used = memory_after - memory_before
        
        print(f"Avg Forward Pass: {avg_time*1000:.2f} ms")
        print(f"Memory Usage: {memory_used:.1f} MB")
        print(f"Throughput: {batch_size/avg_time:.1f} samples/sec")
        
        return {
            'params': param_count,
            'avg_time': avg_time,
            'memory_mb': memory_used,
            'throughput': batch_size/avg_time
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    print("Transformer Configuration Benchmark")
    print("="*60)
    
    configurations = {
        "Ultra-Tiny": {
            "vocab_size": 200,
            "d_model": 32,
            "seq_len": 16,
            "layers": 1,
            "heads": 2,
            "d_ff": 32
        },
        
        "Tiny": {
            "vocab_size": 2000,
            "d_model": 48,
            "seq_len": 16,
            "layers": 1,
            "heads": 2,
            "d_ff": 48
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
            "vocab_size": 10000,
            "d_model": 128,
            "seq_len": 64,
            "layers": 4,
            "heads": 8,
            "d_ff": 256
        }
    }
    
    results = {}
    
    for name, config in configurations.items():
        result = benchmark_config(name, config)
        if result:
            results[name] = result
    
    # Performance comparison
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Config':<12} {'Params':<12} {'Time(ms)':<10} {'Memory(MB)':<12} {'Throughput':<12}")
    print("-" * 80)
    
    for name, data in results.items():
        print(f"{name:<12} {data['params']:>10,} {data['avg_time']*1000:>8.1f} {data['memory_mb']:>10.1f} {data['throughput']:>10.1f}")
    
    # Efficiency analysis
    print(f"\n{'='*80}")
    print("EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    
    if len(results) > 1:
        # Compare smallest vs largest
        sorted_by_params = sorted(results.items(), key=lambda x: x[1]['params'])
        smallest = sorted_by_params[0]
        largest = sorted_by_params[-1]
        
        print(f"\nScaling from {smallest[0]} to {largest[0]}:")
        param_ratio = largest[1]['params'] / smallest[1]['params']
        time_ratio = largest[1]['avg_time'] / smallest[1]['avg_time']
        memory_ratio = largest[1]['memory_mb'] / smallest[1]['memory_mb'] if smallest[1]['memory_mb'] > 0 else 1
        
        print(f"  Parameter increase: {param_ratio:.1f}x")
        print(f"  Time increase: {time_ratio:.1f}x") 
        print(f"  Memory increase: {memory_ratio:.1f}x")
        print(f"  Efficiency ratio (params/time): {param_ratio/time_ratio:.1f}")
        
        # Best efficiency
        efficiencies = {name: data['throughput']/data['params']*1e6 for name, data in results.items()}
        best_efficiency = max(efficiencies.items(), key=lambda x: x[1])
        print(f"\nMost efficient config: {best_efficiency[0]} (throughput/param: {best_efficiency[1]:.3f})")

if __name__ == "__main__":
    main()