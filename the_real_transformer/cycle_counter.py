#!/usr/bin/env python3
"""
Cycle counting utilities for transformer performance measurement
"""

import time
import torch
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Any
import threading

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class CycleCounter:
    """
    High-precision cycle counting for transformer operations
    
    Provides CPU cycle estimates, GPU timing, and memory usage tracking
    """
    
    def __init__(self):
        self.reset()
        self._cpu_freq = self._get_cpu_frequency()
        
    def reset(self):
        """Reset all counters and timers"""
        self.timings = defaultdict(list)
        self.cycle_counts = defaultdict(list) 
        self.memory_usage = defaultdict(list)
        self.operation_counts = defaultdict(int)
        
    def _get_cpu_frequency(self):
        """Get CPU frequency in Hz"""
        if HAS_PSUTIL:
            try:
                # Try to get actual CPU frequency
                freq_info = psutil.cpu_freq()
                if freq_info and freq_info.current:
                    return freq_info.current * 1e6  # Convert MHz to Hz
            except:
                pass
                
        # Fallback to typical modern CPU frequency
        return 3.0e9  # 3 GHz
    
    @contextmanager
    def time_operation(self, operation_name: str, device: str = "cpu"):
        """
        Context manager to time operations and estimate cycles
        
        Args:
            operation_name: Name of the operation being timed
            device: Device type ("cpu" or "cuda")
        """
        self.operation_counts[operation_name] += 1
        
        # Memory usage before
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = psutil.Process().memory_info().rss if HAS_PSUTIL else 0
        
        # Start timing
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            
        try:
            yield
        finally:
            # End timing
            if device == "cuda" and torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
                cpu_time = time.perf_counter() - start_time
                elapsed_time = gpu_time  # Use GPU time as primary
            else:
                elapsed_time = time.perf_counter() - start_time
            
            # Memory usage after
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
            else:
                memory_after = psutil.Process().memory_info().rss if HAS_PSUTIL else 0
            
            # Store measurements
            self.timings[operation_name].append(elapsed_time)
            
            # Estimate CPU cycles (rough approximation)
            estimated_cycles = int(elapsed_time * self._cpu_freq)
            self.cycle_counts[operation_name].append(estimated_cycles)
            
            # Memory delta
            memory_delta = memory_after - memory_before
            self.memory_usage[operation_name].append(memory_delta)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive timing statistics"""
        stats = {}
        
        for operation in self.timings:
            times = self.timings[operation]
            cycles = self.cycle_counts[operation]
            memory = self.memory_usage[operation]
            count = self.operation_counts[operation]
            
            if times:
                stats[operation] = {
                    'count': count,
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_cycles': sum(cycles),
                    'avg_cycles': sum(cycles) / len(cycles),
                    'min_cycles': min(cycles),
                    'max_cycles': max(cycles),
                    'avg_memory_delta': sum(memory) / len(memory) if memory else 0,
                    'times_list': times,
                    'cycles_list': cycles
                }
        
        return stats
    
    def print_summary(self, title="Performance Summary"):
        """Print formatted performance summary"""
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print(f"{'='*60}")
        print(f"CPU Frequency: {self._cpu_freq/1e9:.1f} GHz")
        print()
        
        stats = self.get_stats()
        
        if not stats:
            print("No timing data collected")
            return
            
        # Sort by total time
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"{'Operation':<25} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12} {'Avg Cycles':<15} {'Memory (MB)':<12}")
        print("-" * 100)
        
        total_time = 0
        total_cycles = 0
        
        for operation, data in sorted_ops:
            total_time += data['total_time']
            total_cycles += data['total_cycles']
            
            print(f"{operation:<25} {data['count']:<8} "
                  f"{data['total_time']*1000:<12.2f} "
                  f"{data['avg_time']*1000:<12.2f} "
                  f"{data['avg_cycles']:<15,.0f} "
                  f"{data['avg_memory_delta']/1024/1024:<12.2f}")
        
        print("-" * 100)
        print(f"{'TOTAL':<25} {'':<8} "
              f"{total_time*1000:<12.2f} "
              f"{'':<12} "
              f"{total_cycles:<15,.0f} "
              f"{'':<12}")
        
        print(f"\nTotal inference time: {total_time*1000:.2f} ms")
        print(f"Total estimated cycles: {total_cycles:,.0f}")
        print(f"Equivalent clock cycles at {self._cpu_freq/1e9:.1f} GHz")


# Global counter instance
cycle_counter = CycleCounter()


def count_transformer_cycles(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device="cpu"):
    """
    Detailed cycle counting for transformer inference
    
    Args:
        model: Transformer model
        encoder_input: Source sequence tensor
        encoder_mask: Source attention mask
        decoder_input: Target sequence tensor  
        decoder_mask: Target attention mask
        device: Device to run on
        
    Returns:
        Dictionary with detailed timing and cycle information
    """
    
    cycle_counter.reset()
    model.eval()
    
    # Move inputs to device
    encoder_input = encoder_input.to(device)
    encoder_mask = encoder_mask.to(device)
    decoder_input = decoder_input.to(device)
    decoder_mask = decoder_mask.to(device)
    
    with torch.no_grad():
        # Time the full forward pass
        with cycle_counter.time_operation("full_forward_pass", device):
            # Encoder
            with cycle_counter.time_operation("encoder", device):
                encoder_output = model.encode(encoder_input, encoder_mask)
            
            # Decoder 
            with cycle_counter.time_operation("decoder", device):
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            # Projection
            with cycle_counter.time_operation("projection", device):
                output = model.project(decoder_output)
    
    return cycle_counter.get_stats()


def detailed_layer_profiling(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device="cpu"):
    """
    Profile individual layers and components
    """
    cycle_counter.reset()
    model.eval()
    
    # Move to device
    encoder_input = encoder_input.to(device)
    encoder_mask = encoder_mask.to(device) 
    decoder_input = decoder_input.to(device)
    decoder_mask = decoder_mask.to(device)
    
    with torch.no_grad():
        # Source embeddings
        with cycle_counter.time_operation("src_embedding", device):
            src_embed = model.src_embed(encoder_input)
            
        with cycle_counter.time_operation("src_positional", device):
            src_pos = model.src_pos(src_embed)
        
        # Target embeddings
        with cycle_counter.time_operation("tgt_embedding", device):
            tgt_embed = model.tgt_embed(decoder_input)
            
        with cycle_counter.time_operation("tgt_positional", device):
            tgt_pos = model.tgt_pos(tgt_embed)
        
        # Encoder layers
        encoder_out = src_pos
        for i, layer in enumerate(model.encoder.layers):
            with cycle_counter.time_operation(f"encoder_layer_{i}", device):
                encoder_out = layer(encoder_out, encoder_mask)
        
        # Decoder layers  
        decoder_out = tgt_pos
        for i, layer in enumerate(model.decoder.layers):
            with cycle_counter.time_operation(f"decoder_layer_{i}", device):
                decoder_out = layer(decoder_out, encoder_out, encoder_mask, decoder_mask)
        
        # Final projection
        with cycle_counter.time_operation("final_projection", device):
            output = model.projection_layer(decoder_out)
    
    return cycle_counter.get_stats()


def benchmark_model_sizes(vocab_sizes=[100, 200, 500, 1000], d_models=[32, 48, 64], seq_lens=[16, 32, 64]):
    """
    Benchmark different model configurations
    """
    from model import build_transformer
    
    results = []
    
    for vocab_size in vocab_sizes:
        for d_model in d_models:
            for seq_len in seq_lens:
                if d_model % 2 != 0:  # Must be divisible by num_heads=2
                    continue
                    
                try:
                    print(f"\n🔄 Testing vocab={vocab_size}, d_model={d_model}, seq_len={seq_len}")
                    
                    # Build model
                    model = build_transformer(
                        vocab_size, vocab_size, seq_len, seq_len,
                        d_model=d_model, N=1, h=2, d_ff=d_model
                    )
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    
                    # Create test input
                    batch_size = 1
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)
                    
                    enc_input = torch.randint(0, vocab_size, (batch_size, seq_len))
                    enc_mask = torch.ones(batch_size, 1, 1, seq_len)
                    dec_input = torch.randint(0, vocab_size, (batch_size, seq_len))
                    dec_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
                    
                    # Benchmark
                    stats = count_transformer_cycles(model, enc_input, enc_mask, dec_input, dec_mask, device)
                    
                    total_time = sum(data['total_time'] for data in stats.values())
                    total_cycles = sum(data['total_cycles'] for data in stats.values())
                    
                    results.append({
                        'vocab_size': vocab_size,
                        'd_model': d_model, 
                        'seq_len': seq_len,
                        'parameters': total_params,
                        'time_ms': total_time * 1000,
                        'cycles': total_cycles,
                        'cycles_per_token': total_cycles / seq_len if seq_len > 0 else 0
                    })
                    
                    print(f"   Parameters: {total_params:,}")
                    print(f"   Time: {total_time*1000:.2f} ms")
                    print(f"   Cycles: {total_cycles:,.0f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
    
    return results


if __name__ == "__main__":
    print("🔧 Cycle Counter Utilities")
    print("This module provides cycle counting for transformer performance analysis")
    print()
    print("Usage:")
    print("  from cycle_counter import count_transformer_cycles, cycle_counter")
    print("  stats = count_transformer_cycles(model, enc_input, enc_mask, dec_input, dec_mask)")
    print("  cycle_counter.print_summary()")