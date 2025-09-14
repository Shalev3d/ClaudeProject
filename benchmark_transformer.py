#!/usr/bin/env python3
"""
Transformer FPGA vs CPU Performance Benchmark
Compares training performance between CPU-only and K5 FPGA-accelerated modes
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import argparse
from tqdm import tqdm

from config import get_config
from train import get_ds, get_model
from k5_fpga_accelerator import K5FPGAAccelerator


def benchmark_single_batch(model, batch, device, mode_name):
    """Benchmark single training batch"""
    model.train()
    
    encoder_input = batch['encoder_input'].to(device)
    decoder_input = batch['decoder_input'].to(device) 
    encoder_mask = batch['encoder_mask'].to(device)
    decoder_mask = batch['decoder_mask'].to(device)
    label = batch['label'].to(device)
    
    # Setup loss function and optimizer  
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    start_time = time.time()
    
    # Forward pass
    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
    proj_output = model.project(decoder_output)
    
    # Compute loss
    loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    end_time = time.time()
    batch_time = end_time - start_time
    
    return {
        'mode': mode_name,
        'batch_time': batch_time,
        'loss': loss.item(),
        'batch_size': encoder_input.size(0),
        'seq_len': encoder_input.size(1)
    }


def run_benchmark(num_batches=10, compare_modes=True):
    """Run comprehensive transformer benchmark"""
    print("🚀 Transformer FPGA vs CPU Benchmark")
    print("=" * 50)
    
    # Get configuration with small batch size for fair comparison
    config = get_config()
    config['batch_size'] = 2  # Small batch size for detailed timing
    config['num_epochs'] = 1
    config['max_train_samples'] = 100  # Limit samples for quick testing
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get dataset
    train_dataloader, _, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    results = []
    
    if compare_modes:
        # Test both CPU and FPGA modes
        modes = [
            ('CPU Only', None),
            ('K5 FPGA', K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier"))
        ]
    else:
        # Test only FPGA mode
        modes = [('K5 FPGA', K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier"))]
    
    for mode_name, fpga_accelerator in modes:
        print(f"\n🧪 Testing {mode_name} Mode")
        print("-" * 30)
        
        # Create model for this mode
        model = get_model(config, tokenizer_src.get_vocab_size(), 
                         tokenizer_tgt.get_vocab_size(), fpga_accelerator).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        batch_times = []
        batch_losses = []
        
        # Benchmark multiple batches
        batch_iter = iter(train_dataloader)
        for i in tqdm(range(min(num_batches, len(train_dataloader))), 
                     desc=f"Benchmarking {mode_name}"):
            try:
                batch = next(batch_iter)
                result = benchmark_single_batch(model, batch, device, mode_name)
                batch_times.append(result['batch_time'])
                batch_losses.append(result['loss'])
                
            except StopIteration:
                print(f"Ran out of batches at {i}")
                break
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            avg_loss = sum(batch_losses) / len(batch_losses)
            total_time = sum(batch_times)
            throughput = config['batch_size'] * config['seq_len'] / avg_time
            
            result_summary = {
                'mode': mode_name,
                'batches_tested': len(batch_times),
                'avg_batch_time': avg_time,
                'total_time': total_time,
                'avg_loss': avg_loss,
                'throughput_tokens_per_sec': throughput,
                'batch_size': config['batch_size'],
                'seq_len': config['seq_len']
            }
            
            results.append(result_summary)
            
            print(f"✅ {mode_name} Results:")
            print(f"   • Batches: {len(batch_times)}")
            print(f"   • Avg time/batch: {avg_time*1000:.1f} ms")
            print(f"   • Total time: {total_time:.2f} seconds")
            print(f"   • Avg loss: {avg_loss:.4f}")
            print(f"   • Throughput: {throughput:.0f} tokens/sec")
            
            # Get FPGA statistics if available
            if fpga_accelerator:
                stats = fpga_accelerator.get_performance_stats()
                print(f"   • FPGA operations: {stats['fpga_calls']:,}")
                print(f"   • CPU fallbacks: {stats['cpu_fallback_calls']:,}")
                print(f"   • FPGA usage: {stats['fpga_usage_ratio']:.1%}")
                print(f"   • Avg FPGA time: {stats['fpga_time_average']*1000:.2f} ms")
    
    # Comparison summary
    if len(results) >= 2:
        cpu_result = next(r for r in results if 'CPU' in r['mode'])
        fpga_result = next(r for r in results if 'FPGA' in r['mode'])
        
        speedup = cpu_result['avg_batch_time'] / fpga_result['avg_batch_time']
        throughput_improvement = fpga_result['throughput_tokens_per_sec'] / cpu_result['throughput_tokens_per_sec']
        
        print(f"\n🏆 Performance Comparison Summary")
        print("=" * 50)
        print(f"CPU avg batch time:    {cpu_result['avg_batch_time']*1000:.1f} ms")
        print(f"FPGA avg batch time:   {fpga_result['avg_batch_time']*1000:.1f} ms")
        print(f"Speedup:              {speedup:.2f}x {'🚀' if speedup > 1 else '📉'}")
        print(f"Throughput improvement: {throughput_improvement:.2f}x")
        print(f"CPU throughput:       {cpu_result['throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"FPGA throughput:      {fpga_result['throughput_tokens_per_sec']:.0f} tokens/sec")
        
        if speedup > 1:
            print(f"\n🎉 FPGA acceleration provides {speedup:.1f}x speedup!")
        elif speedup > 0.8:
            print(f"\n⚖️  Performance is comparable (within 20%)")
        else:
            print(f"\n💻 CPU is faster for this configuration")
    
    print("\n" + "=" * 50)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark transformer FPGA acceleration')
    parser.add_argument('--batches', type=int, default=10, 
                       help='Number of batches to benchmark (default: 10)')
    parser.add_argument('--compare', action='store_true', default=True,
                       help='Compare CPU vs FPGA performance (default: True)')
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark(num_batches=args.batches, compare_modes=args.compare)
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        raise