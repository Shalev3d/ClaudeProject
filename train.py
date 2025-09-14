from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from k5_fpga_accelerator import K5FPGAAccelerator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter



os.environ["TOKENIZERS_PARALLELISM"] = "false"   # HF tokenizers can SIGBUS on mac if parallel
os.environ["OMP_NUM_THREADS"] = "1"              # reduce thread contention
os.environ["MKL_NUM_THREADS"] = "1"

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer,
                   num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# from huginface
def get_or_build_tokenizer(config, ds, lang):
    # Use separate tokenizer files for reduced vocabulary
    if config.get('reduced_vocab', False):
        tokenizer_path = Path(f"tokenizer_{lang}_reduced_{config['vocab_size']}.json")
    else:
        tokenizer_path = Path(config['tokenizer_file'].format(lang))
        
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Configure vocabulary size limit
        if config.get('reduced_vocab', False):
            # Limit vocabulary to specified size (minus 4 special tokens)
            vocab_limit = config['vocab_size'] - 4  
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
                min_frequency=2,
                vocab_size=vocab_limit
            )
            print(f"Training tokenizer with reduced vocabulary of {config['vocab_size']} words (including special tokens)")
        else:
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        
        if config.get('reduced_vocab', False):
            print(f"Saved reduced vocabulary tokenizer with {tokenizer.get_vocab_size()} tokens")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if config.get('reduced_vocab', False):
            print(f"Loaded reduced vocabulary tokenizer with {tokenizer.get_vocab_size()} tokens")
            
    return tokenizer


def get_ds(config):
    # It only has the train split, so we divide it overselves
    # Load dataset with configurable sample limit
    if config.get('max_train_samples') is not None:
        split_str = f"train[:{config['max_train_samples']}]"
    else:
        split_str = 'train'
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split=split_str)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],
                              config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len, fpga_accelerator=None):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
                              d_model=config['d_model'], N=config['layers'], h=config['heads'],
                              d_ff=config.get('d_ff', config['d_model'] * 4), 
                              fpga_accelerator=fpga_accelerator)
    return model

def perform_epoch_cycle_analysis(model, val_dataloader, device, epoch, config):
    """Perform cycle analysis after each epoch"""
    try:
        print(f"\n⏱️  EPOCH {epoch + 1} CYCLE ANALYSIS")
        print("="*50)
        
        from cycle_counter import count_transformer_cycles, cycle_counter
        
        # Get a sample batch from validation data
        sample_batch = next(iter(val_dataloader))
        encoder_input = sample_batch["encoder_input"][:1].to(device)  # Take first sample only
        encoder_mask = sample_batch["encoder_mask"][:1].to(device)
        decoder_input = sample_batch["decoder_input"][:1].to(device) 
        decoder_mask = sample_batch["decoder_mask"][:1].to(device)
        
        # Set model to evaluation mode for consistent timing
        model.eval()
        
        # Count cycles
        with torch.no_grad():
            stats = count_transformer_cycles(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device)
        
        # Calculate totals
        total_time = sum(data['total_time'] for data in stats.values())
        total_cycles = sum(data['total_cycles'] for data in stats.values())
        
        # Estimate cycles for full epoch
        num_batches = len(val_dataloader)
        epoch_time_estimate = total_time * num_batches
        epoch_cycles_estimate = total_cycles * num_batches
        
        print(f"Single inference:")
        print(f"  • Time: {total_time*1000:.1f} ms")
        print(f"  • Cycles: {total_cycles:,.0f}")
        print(f"  • Sequence length: {config['seq_len']}")
        print(f"  • Cycles per token: {total_cycles//config['seq_len']:,.0f}")
        
        print(f"\\nFull epoch estimate ({num_batches} batches):")
        print(f"  • Total time: {epoch_time_estimate:.1f} seconds")
        print(f"  • Total cycles: {epoch_cycles_estimate:,.0f}")
        print(f"  • Minutes: {epoch_time_estimate/60:.1f}")
        
        # FPGA analysis
        fpga_clock_freq = 100e6  # 100 MHz
        fpga_cycles_single = total_cycles * (cycle_counter._cpu_freq / fpga_clock_freq)
        fpga_time_single = fpga_cycles_single / fpga_clock_freq
        fpga_epoch_time = fpga_time_single * num_batches
        
        print(f"\\nFPGA estimates (100MHz):")
        print(f"  • Single inference FPGA time: {fpga_time_single*1000:.1f} ms")
        print(f"  • Single inference speedup: {total_time/fpga_time_single:.2f}x")
        print(f"  • Full epoch FPGA time: {fpga_epoch_time:.1f} seconds")
        print(f"  • Full epoch speedup: {epoch_time_estimate/fpga_epoch_time:.2f}x")
        
        if total_time/fpga_time_single > 1.0:
            print(f"  🚀 FPGA would be FASTER for this epoch!")
        else:
            print(f"  💻 CPU remains faster for this model size")
        
        # Set model back to training mode
        model.train()
        
        print("="*50)
        
    except Exception as e:
        print(f"⚠️  Cycle analysis failed: {e}")
        # Set model back to training mode even if analysis fails
        model.train()


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"# if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Initialize K5 FPGA accelerator if available
    fpga_accelerator = None
    k5_app_name = config.get('k5_app_name', 'de10_lite_matrix_multiplier')  # K5 application name
    
    if config.get('use_fpga', False):
        try:
            print(f"Initializing K5 FPGA accelerator with app '{k5_app_name}'...")
            fpga_accelerator = K5FPGAAccelerator(k5_app_name=k5_app_name)
            print("K5 FPGA accelerator initialized successfully!")
            print("Matrix multiplications will be offloaded to FPGA via K5 system")
        except Exception as e:
            print(f"K5 FPGA initialization failed: {e}")
            print("Continuing with CPU/GPU only")
            fpga_accelerator = None

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), fpga_accelerator).to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"\n🏗️  Model Information:")
    print(f"   • Source vocabulary: {tokenizer_src.get_vocab_size():,} tokens")
    print(f"   • Target vocabulary: {tokenizer_tgt.get_vocab_size():,} tokens") 
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Model size: {model_size_mb:.1f} MB")
    print(f"   • Training samples: {config.get('max_train_samples', 'All'):,}")
    print(f"   • Device: {device}")
    
    # Perform cycle analysis like in test_reduced_vocab.py
    try:
        print(f"\n⏱️  Performance Analysis:")
        from cycle_counter import count_transformer_cycles, cycle_counter
        
        # Create a sample batch for testing
        sample_batch = next(iter(train_dataloader))
        encoder_input = sample_batch["encoder_input"][:1].to(device)  # Take first sample only
        encoder_mask = sample_batch["encoder_mask"][:1].to(device)
        decoder_input = sample_batch["decoder_input"][:1].to(device)
        decoder_mask = sample_batch["decoder_mask"][:1].to(device)
        
        # Count cycles
        model.eval()
        with torch.no_grad():
            stats = count_transformer_cycles(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device)
        
        # Calculate totals
        total_time = sum(data['total_time'] for data in stats.values())
        total_cycles = sum(data['total_cycles'] for data in stats.values())
        seq_len = config['seq_len']
        
        print(f"   • Inference time: {total_time*1000:.2f} ms")
        print(f"   • Estimated cycles: {total_cycles:,.0f}")
        print(f"   • Cycles per token: {total_cycles//seq_len:,.0f}")
        print(f"   • Tokens per second: {seq_len/total_time:.0f}")
        
        # FPGA estimation
        fpga_clock_freq = 100e6  # 100 MHz
        fpga_cycles = total_cycles * (cycle_counter._cpu_freq / fpga_clock_freq)
        fpga_time = fpga_cycles / fpga_clock_freq
        
        print(f"   • FPGA cycles (100MHz): {fpga_cycles:,.0f}")
        print(f"   • FPGA time estimate: {fpga_time*1000:.2f} ms")
        print(f"   • FPGA speedup potential: {total_time/fpga_time:.1f}x")
        
        model.train()  # Switch back to training mode
        
    except Exception as e:
        print(f"   • Cycle analysis failed: {e}")
    
    print()
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,
                                                                                                        preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)

        # Perform cycle analysis after each epoch
        perform_epoch_cycle_analysis(model, val_dataloader, device, epoch, config)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    # Print FPGA performance statistics if used
    if fpga_accelerator:
        stats = fpga_accelerator.get_performance_stats()
        print("\n🚀 K5 FPGA Performance Summary:")
        print(f"   • FPGA operations: {stats['fpga_calls']:,}")
        print(f"   • CPU fallbacks: {stats['cpu_fallback_calls']:,}")
        print(f"   • FPGA usage ratio: {stats['fpga_usage_ratio']:.1%}")
        print(f"   • Average FPGA time: {stats['fpga_time_average']*1000:.2f} ms")
        print(f"   • Total FPGA time: {stats['fpga_time_total']:.2f} seconds")
        print("K5 FPGA accelerator session completed")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
