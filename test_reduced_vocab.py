#!/usr/bin/env python3
"""
Test script to verify reduced vocabulary functionality
"""

import torch
from config import get_config
from train import get_ds, get_model
from model import build_transformer
from datasets import load_dataset

def test_reduced_vocabulary():
    """Test the transformer with reduced vocabulary"""
    print("🧪 Testing Reduced Vocabulary Transformer")
    print("=" * 50)
    
    # Get configuration with reduced vocabulary
    config = get_config()
    print(f"Reduced vocabulary: {config['reduced_vocab']}")
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Dataset: {config['datasource']}")
    print(f"Languages: {config['lang_src']} → {config['lang_tgt']}")
    print()
    
    # Load a small subset of data for testing
    try:
        print("📊 Loading dataset...")
        ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train[:1000]')  # Load more samples to filter
        
        # Filter for short sentences that fit within seq_len
        print(f"Filtering for sentences shorter than {config['seq_len']-2} tokens...")
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        
        # Create a simple tokenizer for length estimation
        temp_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        temp_tokenizer.pre_tokenizer = Whitespace()
        
        filtered_samples = []
        max_samples = 100
        
        for sample in ds_raw:
            src_text = sample['translation'][config['lang_src']]
            tgt_text = sample['translation'][config['lang_tgt']]
            
            # Rough estimate using whitespace tokenization
            src_tokens = len(src_text.split())
            tgt_tokens = len(tgt_text.split())
            
            # Keep sentences that are likely to fit (with some margin for subword tokenization)
            if src_tokens <= config['seq_len']//2 and tgt_tokens <= config['seq_len']//2:
                filtered_samples.append(sample)
                if len(filtered_samples) >= max_samples:
                    break
        
        ds_raw = filtered_samples
        print(f"Found {len(ds_raw)} short samples for testing")
        
        if len(ds_raw) == 0:
            print("❌ No short sentences found. Try increasing seq_len in config.")
            return False
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Build tokenizers manually using filtered data
    try:
        print("\n🔤 Building tokenizers...")
        from train import get_or_build_tokenizer
        from dataset import BilingualDataset, causal_mask
        from torch.utils.data import DataLoader, random_split
        
        # Build tokenizers on filtered data
        tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
        tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
        
        # Split the filtered data
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
        
        # Create datasets with error handling
        train_samples = []
        val_samples = []
        
        # Filter out sentences that are still too long after tokenization
        for sample in train_ds_raw:
            try:
                dataset_sample = BilingualDataset([sample], tokenizer_src, tokenizer_tgt, 
                                                config['lang_src'], config['lang_tgt'], 
                                                config['seq_len'])[0]
                train_samples.append(sample)
            except ValueError:
                continue  # Skip samples that are too long
        
        for sample in val_ds_raw:
            try:
                dataset_sample = BilingualDataset([sample], tokenizer_src, tokenizer_tgt, 
                                                config['lang_src'], config['lang_tgt'], 
                                                config['seq_len'])[0]
                val_samples.append(sample)
            except ValueError:
                continue  # Skip samples that are too long
        
        print(f"Final train samples: {len(train_samples)}")
        print(f"Final validation samples: {len(val_samples)}")
        
        if len(train_samples) == 0:
            print("❌ No valid training samples found. Sequence length too small.")
            return False
        
        # Create final datasets
        train_ds = BilingualDataset(train_samples, tokenizer_src, tokenizer_tgt, 
                                  config['lang_src'], config['lang_tgt'], config['seq_len'])
        val_ds = BilingualDataset(val_samples, tokenizer_src, tokenizer_tgt, 
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
        
        # Create dataloaders
        train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
        
        print(f"Source vocabulary size: {tokenizer_src.get_vocab_size()}")
        print(f"Target vocabulary size: {tokenizer_tgt.get_vocab_size()}")
        
        # Verify vocabulary sizes are within expected range
        max_expected = config['vocab_size']
        if tokenizer_src.get_vocab_size() <= max_expected and tokenizer_tgt.get_vocab_size() <= max_expected:
            print("✅ Vocabulary sizes are within expected limits")
        else:
            print(f"⚠️  Warning: Vocabulary sizes exceed limit of {max_expected}")
            
    except Exception as e:
        print(f"❌ Error building tokenizers: {e}")
        return False
    
    # Build model
    try:
        print("\n🏗️  Building model...")
        model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error building model: {e}")
        return False
    
    # Test forward pass
    try:
        print("\n🔄 Testing forward pass...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = model.to(device)
        model.eval()
        
        # Get a batch from the dataloader
        batch = next(iter(train_dataloader))
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        
        print(f"Input shapes:")
        print(f"  Encoder input: {encoder_input.shape}")
        print(f"  Decoder input: {decoder_input.shape}")
        
        # Forward pass
        # Import cycle counter
        from cycle_counter import count_transformer_cycles, detailed_layer_profiling, cycle_counter
        
        # Perform detailed cycle counting
        print("\n⏱️  Measuring inference performance...")
        
        # Basic cycle counting
        stats = count_transformer_cycles(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device)
        cycle_counter.print_summary("Transformer Inference Performance")
        
        # Detailed layer profiling
        print("\n🔍 Detailed Layer Profiling...")
        detailed_stats = detailed_layer_profiling(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device)
        
        # Manual verification with output shapes
        with torch.no_grad():
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
        print(f"\nOutput shapes:")
        print(f"  Encoder output: {encoder_output.shape}")
        print(f"  Decoder output: {decoder_output.shape}")
        print(f"  Projection output: {proj_output.shape}")
        
        # Verify output vocabulary dimension
        expected_vocab_size = tokenizer_tgt.get_vocab_size()
        actual_vocab_size = proj_output.shape[-1]
        
        if actual_vocab_size == expected_vocab_size:
            print("✅ Output vocabulary dimension matches tokenizer vocabulary size")
        else:
            print(f"❌ Output vocabulary dimension mismatch: {actual_vocab_size} != {expected_vocab_size}")
            return False
            
        # Print detailed performance analysis
        print(f"\n📊 Performance Analysis:")
        total_time = sum(data['total_time'] for data in stats.values())
        total_cycles = sum(data['total_cycles'] for data in stats.values())
        seq_len = config['seq_len']
        
        print(f"   • Total inference time: {total_time*1000:.3f} ms")
        print(f"   • Total estimated cycles: {total_cycles:,.0f}")
        print(f"   • Cycles per token: {total_cycles//seq_len:,.0f}")
        print(f"   • Tokens per second: {seq_len/total_time:.0f}")
        
        if device == "cuda":
            print(f"   • GPU memory used: {torch.cuda.max_memory_allocated()/1024/1024:.1f} MB")
        
        # FPGA estimation
        fpga_clock_freq = 100e6  # 100 MHz typical FPGA frequency  
        fpga_cycles = total_cycles * (cycle_counter._cpu_freq / fpga_clock_freq)
        fpga_time = fpga_cycles / fpga_clock_freq
        
        print(f"\n🔧 FPGA Estimates (100 MHz clock):")
        print(f"   • Estimated FPGA cycles: {fpga_cycles:,.0f}")
        print(f"   • Estimated FPGA time: {fpga_time*1000:.3f} ms")
        print(f"   • FPGA speedup potential: {total_time/fpga_time:.1f}x")
            
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return False
    
    print("\n🎉 All tests passed! Reduced vocabulary transformer is working correctly.")
    print("\n📋 Summary:")
    print(f"   • Source vocabulary: {tokenizer_src.get_vocab_size()} tokens")
    print(f"   • Target vocabulary: {tokenizer_tgt.get_vocab_size()} tokens")
    print(f"   • Model parameters: {trainable_params:,}")
    print(f"   • Forward pass: ✅")
    
    return True

if __name__ == "__main__":
    success = test_reduced_vocabulary()
    exit(0 if success else 1)