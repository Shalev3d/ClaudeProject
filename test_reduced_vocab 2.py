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
        ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train[:100]')  # Only first 100 samples
        print(f"Loaded {len(ds_raw)} samples for testing")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Build tokenizers and dataset
    try:
        print("\n🔤 Building tokenizers...")
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
        
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
        with torch.no_grad():
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
        print(f"Output shapes:")
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