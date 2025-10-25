import torch
import time
from statistics import mean
from typing import Callable, Dict, List
import yaml
from dataclasses import dataclass
import os
import argparse
from cs336_basics.model import BasicsTransformerLM # type: ignore
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import cross_entropy
import timeit
@dataclass
class TrainingConfig:
    # Model params
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    vocab_size: int
    context_length: int
    rope_theta: float
    
    # Training params
    batch_size: int
    learning_rate: float
    min_lr: float
    warmup_iters: int
    max_tokens: int
    weight_decay: float
    betas: tuple[float, float]
    eps: float 
    gradient_clip: float
    
    # Data paths
    train_file: str
    valid_file: str
    checkpoint_dir: str
    tokenizer_path: str
    
    # Training control
    eval_interval: int
    save_interval: int
    log_interval: int
    
    # Logging
    project_name: str
    experiment_name: str


def get_training_config(config_path: str = 'cs336_systems/config.yaml') -> TrainingConfig:
    """
    Load and parse training configuration from YAML file with CLI overrides.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        TrainingConfig object with all parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or YAML parsing fails
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load and parse YAML config
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")
    
    # Setup argument parser for CLI overrides
    parser = argparse.ArgumentParser(description="Transformer Language Model Training")
    
    # Config file argument
    parser.add_argument("--config", type=str, default='cs336_basics/config.yaml', help="Path to config file")
    
    # Model arguments that can be overridden
    parser.add_argument("--d_model", type=int, help="Override model dimension")
    parser.add_argument("--num_heads", type=int, help="Override number of attention heads")
    parser.add_argument("--num_layers", type=int, help="Override number of transformer layers")
    parser.add_argument("--d_ff", type=int, help="Override feedforward dimension")
    
    # Training arguments that can be overridden
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--max_tokens", type=int, help="Override maximum tokens processed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Flatten nested config structure
    flattened_config = {}
    
    # Model parameters
    if 'model' in config:
        flattened_config.update(config['model'])
    
    # Training parameters
    if 'training' in config:
        flattened_config.update(config['training'])
    
    # Data parameters
    if 'data' in config:
        flattened_config.update(config['data'])
    
    # Logging parameters
    if 'logging' in config:
        flattened_config.update(config['logging'])
    
    # Required config fields
    required_fields = [
        'd_model', 'num_heads', 'num_layers', 'd_ff',
        'vocab_size', 'context_length', 'rope_theta',
        'batch_size', 'learning_rate', 'min_lr', 'warmup_iters', 'max_tokens',
        'weight_decay', 'betas', 'eps', 'gradient_clip',
        'train_file', 'valid_file', 'checkpoint_dir', 'tokenizer_path',
        'eval_interval', 'save_interval', 'log_interval',
        'project_name', 'experiment_name'
    ]
    
    # Validate required fields
    for field in required_fields:
        if field not in flattened_config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Update config with CLI overrides
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None and key in flattened_config:
            flattened_config[key] = value
            print(f"Overriding {key} with CLI value: {value}")
    
    # Convert data types to match TrainingConfig requirements
    # Convert betas list to tuple
    if 'betas' in flattened_config and isinstance(flattened_config['betas'], list):
        flattened_config['betas'] = tuple(flattened_config['betas'])
    
    # Convert numeric strings to appropriate types
    numeric_fields = [
        'd_model', 'num_heads', 'num_layers', 'd_ff', 'vocab_size', 'context_length',
        'batch_size', 'warmup_iters', 'max_tokens', 'eval_interval', 'save_interval', 'log_interval'
    ]
    
    float_fields = [
        'learning_rate', 'min_lr', 'weight_decay', 'eps', 'gradient_clip', 'rope_theta'
    ]
    
    for field in numeric_fields:
        if field in flattened_config:
            flattened_config[field] = int(flattened_config[field])
    
    for field in float_fields:
        if field in flattened_config:
            flattened_config[field] = float(flattened_config[field])
    
    # Convert to TrainingConfig object
    return TrainingConfig(**flattened_config)



def benchmark(description: str, fn: callable, num_warmups: int=1, num_trials: int =3):
    # warmup run
    print("in test")
    for _ in range(num_warmups):
        fn()
    print("warm up finished")
    if torch.cuda.is_available():
        torch.cuda.synchronize() #wait for Cuda threads to finish
    times=[]
    # time for num_trials
    for _ in range(num_trials):   
        start_time =  timeit.default_timer()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize() #wait for Cuda threads to finish
        end_time =  timeit.default_timer()
        times.append((end_time-start_time)*1000)
    # average computation
    mean_time =  mean(times)
    #result printing
    print(f"\nBenchmark:{description}")
    print(f":{mean_time:.3f} ms")
    return mean_time
            
    
def main():
    config = get_training_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    # model initialization
    model = BasicsTransformerLM (
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        context_length=config.context_length,
        rope_theta=config.rope_theta
    ).to(device)
     # optimizer, learning rate set up
    optimizer = AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay=config.weight_decay,
        betas= config.betas,
        eps = config.eps
    )
    # random input and label
    random_batch  =  torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device = device)
    random_label = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device = device)
    loss = None
    def forward_run():
        out = model(random_batch)
        loss = cross_entropy(out, random_label)
    def forward_backward_run():
        out = model(random_batch)
        loss = cross_entropy(out, random_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    run = forward_backward_run
    print("start benchmarking")
    benchmark(description ="model forward and backward run", fn = run, num_warmups =1, num_trials = 2)

if __name__ == "__main__":
    main()

    


    
    


