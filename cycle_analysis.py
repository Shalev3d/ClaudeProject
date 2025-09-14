#!/usr/bin/env python3
"""
Detailed CPU and FPGA Cycle Analysis for Transformer Training
Shows cycle counts for both CPU operations and FPGA operations
"""

import torch
import time
import psutil
from config import get_config
from train import get_ds, get_model
from k5_fpga_accelerator import K5FPGAAccelerator
from cycle_counter import cycle_counter


def detailed_cycle_analysis():
    """Perform detailed cycle analysis showing CPU vs FPGA breakdown"""
    print("🔬 Detailed CPU & FPGA Cycle Analysis")
    print("=" * 60)
    
    # Setup
    config = get_config()
    config['batch_size'] = 1  # Single sample for precise measurement
    config['max_train_samples'] = 10  # Minimal dataset
    
    device = torch.device("cpu")  # Force CPU for consistent measurement
    train_dataloader, _, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Get a sample batch
    sample_batch = next(iter(train_dataloader))
    encoder_input = sample_batch["encoder_input"].to(device)
    encoder_mask = sample_batch["encoder_mask"].to(device)
    decoder_input = sample_batch["decoder_input"].to(device)
    decoder_mask = sample_batch["decoder_mask"].to(device)
    
    print(f"Sequence length: {config['seq_len']}")
    print(f"Model dimensions: d_model={config['d_model']}, heads={config['heads']}")
    print(f"CPU frequency: {psutil.cpu_freq().current:.0f} MHz")
    
    # Test both modes
    modes = [
        ("CPU Only", None),
        ("K5 FPGA", K5FPGAAccelerator(k5_app_name="de10_lite_matrix_multiplier"))
    ]
    
    results = {}
    
    for mode_name, fpga_accelerator in modes:
        print(f"\n📊 {mode_name} Analysis")
        print("-" * 40)
        
        # Create model
        model = get_model(config, tokenizer_src.get_vocab_size(), 
                         tokenizer_tgt.get_vocab_size(), fpga_accelerator).to(device)
        model.eval()
        
        # Reset cycle counter
        cycle_counter.reset()
        
        # Reset FPGA stats if available
        if fpga_accelerator:
            fpga_accelerator.reset_stats()
        
        with torch.no_grad():
            # Forward pass with detailed cycle counting
            with cycle_counter.time_operation("encoder", "cpu"):
                encoder_output = model.encode(encoder_input, encoder_mask)
            
            with cycle_counter.time_operation("decoder", "cpu"):
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            with cycle_counter.time_operation("projection", "cpu"):
                proj_output = model.project(decoder_output)
        
        # Get statistics
        stats = cycle_counter.get_stats()
        total_time = sum(data['total_time'] for data in stats.values())
        total_cycles = sum(data['total_cycles'] for data in stats.values())
        encoder_cycles = stats.get('encoder', {}).get('total_cycles', 0)
        decoder_cycles = stats.get('decoder', {}).get('total_cycles', 0)  
        projection_cycles = stats.get('projection', {}).get('total_cycles', 0)
        
        # Get FPGA statistics
        fpga_stats = fpga_accelerator.get_performance_stats() if fpga_accelerator else None
        
        results[mode_name] = {
            'total_time': total_time,
            'total_cycles': total_cycles,
            'encoder_cycles': encoder_cycles,
            'decoder_cycles': decoder_cycles,
            'projection_cycles': projection_cycles,
            'fpga_stats': fpga_stats
        }
        
        print(f"⏱️  Timing Results:")
        print(f"   • Total time: {total_time*1000:.2f} ms")
        print(f"   • Total CPU cycles: {total_cycles:,}")
        print(f"   • Cycles per token: {total_cycles // config['seq_len']:,}")
        
        print(f"\n🧮 CPU Cycle Breakdown:")
        print(f"   • Encoder: {encoder_cycles:,} cycles ({encoder_cycles/total_cycles:.1%})")
        print(f"   • Decoder: {decoder_cycles:,} cycles ({decoder_cycles/total_cycles:.1%})")
        print(f"   • Projection: {projection_cycles:,} cycles ({projection_cycles/total_cycles:.1%})")
        
        if fpga_stats and fpga_stats['fpga_calls'] > 0:
            print(f"\n🚀 FPGA Statistics:")
            print(f"   • FPGA operations: {fpga_stats['fpga_calls']:,}")
            print(f"   • CPU fallbacks: {fpga_stats['cpu_fallback_calls']:,}")
            print(f"   • FPGA usage ratio: {fpga_stats['fpga_usage_ratio']:.1%}")
            print(f"   • Total FPGA time: {fpga_stats['fpga_time_total']*1000:.2f} ms")
            print(f"   • Average FPGA operation: {fpga_stats['fpga_time_average']*1000:.3f} ms")
            
            # Estimate FPGA cycles (based on communication overhead + computation)
            fpga_clock_freq = 100e6  # 100 MHz FPGA
            cpu_freq = psutil.cpu_freq().current * 1e6  # Convert to Hz
            
            # FPGA cycles for matrix operations (excluding communication)
            fpga_compute_cycles = fpga_stats['fpga_calls'] * 1000  # Rough estimate
            # Communication cycles (much higher due to file I/O)
            fpga_comm_cycles = fpga_stats['fpga_time_total'] * fpga_clock_freq
            
            print(f"   • Estimated FPGA compute cycles: {fpga_compute_cycles:,.0f}")
            print(f"   • Estimated FPGA comm cycles: {fpga_comm_cycles:,.0f}")
        else:
            print(f"\n💻 Pure CPU computation (no FPGA operations)")
    
    # Comparison
    if len(results) >= 2:
        cpu_result = results['CPU Only']
        fpga_result = results['K5 FPGA']
        
        cpu_cycles = cpu_result['total_cycles']
        fpga_cycles = fpga_result['total_cycles']
        
        print(f"\n🏆 Performance Comparison")
        print("=" * 60)
        print(f"CPU-only total cycles:     {cpu_cycles:,}")
        print(f"FPGA-enabled total cycles: {fpga_cycles:,}")
        
        if fpga_cycles < cpu_cycles:
            savings = cpu_cycles - fpga_cycles
            print(f"Cycle savings:            {savings:,} cycles ({savings/cpu_cycles:.1%} reduction)")
        else:
            overhead = fpga_cycles - cpu_cycles  
            print(f"Cycle overhead:           {overhead:,} cycles ({overhead/cpu_cycles:.1%} increase)")
        
        print(f"Performance ratio:        {cpu_cycles/fpga_cycles:.2f}x")
        
        # FPGA specific analysis
        fpga_stats = fpga_result['fpga_stats']
        if fpga_stats and fpga_stats['fpga_calls'] > 0:
            print(f"\n🔍 FPGA Efficiency Analysis:")
            print(f"Matrix operations offloaded: {fpga_stats['fpga_calls']:,}")
            print(f"Time spent in FPGA calls:    {fpga_stats['fpga_time_total']*1000:.1f} ms")
            print(f"FPGA communication overhead dominates due to file I/O")
            print(f"For small matrices, CPU is faster due to setup costs")
        else:
            print(f"\n⚠️  No FPGA operations performed (matrices too large or other constraints)")


def show_fpga_board_cycles():
    """Show how to measure cycles directly on the FPGA board"""
    print(f"\n📟 Measuring Cycles on FPGA Board")
    print("=" * 50)
    
    print("To see actual FPGA board cycles, you need to:")
    print("1. 🔧 Modify the SystemVerilog design to include cycle counters")
    print("2. 📊 Add performance monitoring registers")
    print("3. 🔍 Read cycle counts via UART or memory interface")
    
    print(f"\n💡 Recommended FPGA Cycle Measurement:")
    print("Add this to your matrix_multiplier.sv:")
    
    verilog_code = '''
// Add cycle counter to FPGA design
reg [31:0] cycle_counter;
reg computation_active;

always @(posedge clk) begin
    if (reset) begin
        cycle_counter <= 0;
        computation_active <= 0;
    end else if (start_computation) begin
        cycle_counter <= 0;
        computation_active <= 1;
    end else if (computation_active) begin
        cycle_counter <= cycle_counter + 1;
        if (computation_done) begin
            computation_active <= 0;
            // Send cycle count via UART
            cycle_count_out <= cycle_counter;
        end
    end
end'''
    
    print(verilog_code)
    
    print(f"\n📡 Then modify your C program to read cycle counts:")
    
    c_code = '''
// In de10_lite_multiplier.c, add cycle reading
uint32_t read_fpga_cycles() {
    uint32_t cycles = 0;
    // Read cycle count from FPGA after computation
    // This would be sent via UART along with results
    uart_receive_bytes((uint8_t*)&cycles, 4);
    return cycles;
}'''
    
    print(c_code)
    
    print(f"\n🎯 This would give you:")
    print("• Exact FPGA clock cycles for matrix multiplication")
    print("• Separate cycle counts for different matrix sizes")  
    print("• Hardware-accurate performance measurement")
    print("• Comparison with CPU cycle estimates")


if __name__ == "__main__":
    try:
        detailed_cycle_analysis()
        show_fpga_board_cycles()
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise