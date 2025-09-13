from pathlib import Path

def get_config():
    return {
        "batch_size": 1,            # Small batch for long inference testing
        "num_epochs": 2,            # Fewer epochs due to long computation
        "lr": 10**-4,
        "max_train_samples": 1000,  # Fewer samples due to 512-token sequences
        "seq_len": 1024,            # ULTRA-LONG sequence length (16x longer!)
        "d_model": 16,              # Optimized for maximum inference time
        "datasource": 'Helsinki-NLP/opus-100',
        "lang_src": "en",
        "lang_tgt": "he",
        "model_folder": "weights_max_compute",
        "model_basename": "tmodel_max_",
        "preload": None,  # Don't load previous model
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_max_compute",
        "layers": 6,                # DEEPEST architecture (6 encoder + 6 decoder layers)
        "heads": 4,                 # 4 attention heads
        "d_ff": 16,                 # Matched to d_model
        # Vocabulary settings
        "reduced_vocab": True,  # Set to True to use reduced vocabulary
        "vocab_size": 350,          # Optimized vocabulary size for longest inference
        # FPGA acceleration settings
        "use_fpga": False,  # Set to True to enable FPGA acceleration
        "fpga_port": "/dev/ttyUSB0",  # UART port for FPGA communication
        "fpga_baudrate": 115200,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
