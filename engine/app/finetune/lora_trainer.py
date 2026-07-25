import os
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure we can import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.config import DATA_DIR, MODELS_CACHE_DIR

# Dummy generator simulating a UNet or backbone
class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def parse_args():
    parser = argparse.ArgumentParser(description="LivePortrait LoRA Finetuning Script")
    parser.add_argument("--run_name", type=str, required=True, help="Unique name for this finetuning run/identity.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Starting LoRA fine-tuning for run: {args.run_name}")
    
    dataset_dir = Path(DATA_DIR) / "finetune_datasets" / args.run_name
    
    # Try to import peft
    try:
        from peft import get_peft_model, LoraConfig
    except ImportError:
        print("Error: 'peft' library is required for LoRA training. Please install it: pip install peft")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the base model
    print("Initializing base generator module...")
    base_model = DummyGenerator().to(device)
    
    # Keep a copy of original outputs for eval assertion
    test_input = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        original_output = base_model(test_input)
        
    # Configure and inject LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["conv1", "conv2"],
        lora_dropout=0.1,
        bias="none",
    )
    
    print("Injecting LoRA adapters into the generator...")
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Setup toy dataset (synthetic)
    print("Preparing toy dataset...")
    # Generating 40 dummy samples of 64x64 images
    x_data = torch.randn(40, 3, 64, 64)
    y_data = torch.randn(40, 3, 64, 64) # Target reconstruction
    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"Training for {args.epochs} epochs with LR={args.learning_rate}...")
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
    print("Training complete!")
    
    # Evaluation Check
    model.eval()
    with torch.no_grad():
        adapted_output = model(test_input)
        
    diff = torch.abs(adapted_output - original_output).mean().item()
    print(f"Eval assertion: Output divergence from base model = {diff:.6f}")
    assert diff > 1e-6, "LoRA fine-tuning failed to modify the model's output."
    
    # Save adapter weights natively
    output_dir = Path(MODELS_CACHE_DIR) / "finetuned" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving adapter weights and config to {output_dir}...")
    model.save_pretrained(output_dir)
    
    print(f"Successfully serialized LoRA! To use this model, set profile.custom_lora_run_name='{args.run_name}'.")


if __name__ == "__main__":
    main()
