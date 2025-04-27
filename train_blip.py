import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import Adafactor, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm
import logging
import shutil
from datetime import datetime

# Set up argument parser
parser = argparse.ArgumentParser(description='Train BLIP2 model for image captioning')
parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing all your trainning data')
parser.add_argument('--output_dir', type=str, default='output/blip2_finetuned', help='Directory to save model outputs')
parser.add_argument('--model_name', type=str, default='Salesforce/blip2-flan-t5-xl-coco', help='Base model to finetune')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimization')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--checkpoint_dir', type=str, default='/ephemeral/blip2_train_2', help='Directory to save checkpoints')
parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--eval_samples', type=int, default=50, help='Number of samples for evaluation')
parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
parser.add_argument('--prompt', type=str, 
                    default='Look at the image carefully and describe it in one complete sentence. Focus on the main subject, important actions, and the context or setting.',
                    help='Prompt for image captioning')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blip2_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# For reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class BLIP:
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl-coco")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl-coco").to(self.device)

    def image_captioning(self, img_path: str, text="Look at the image carefully and describe it in one complete sentence. Focus on the main subject, important actions, and the context or setting. Use natural, descriptive language that a human would use when explaining the image to someone who cannot see it.") -> str:
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=200)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            return "Error generating caption"


class ImageCaptioningDataset(Dataset):
    def __init__(self, data_dir, processor, prompt="Look at the image carefully and describe it in one complete sentence."):
        self.data_dir = data_dir
        self.processor = processor
        self.prompt = prompt
        self.samples = []
        
        # Collect all image-caption pairs from all directories
        if isinstance(data_dir, str):
            self._collect_samples_from_dir(data_dir)
        elif isinstance(data_dir, list):
            for dir_path in data_dir:
                self._collect_samples_from_dir(dir_path)
                
        logger.info(f"Total dataset size: {len(self.samples)} image-caption pairs")
        
        # Log sample distribution by directory
        self._log_dataset_distribution()
        
    def _collect_samples_from_dir(self, dir_path):
        """Collect all image-caption pairs from a directory and its subdirectories"""
        start_count = len(self.samples)
        
        # Verify directory exists before attempting to walk it
        if not os.path.exists(dir_path):
            logger.warning(f"Directory {dir_path} does not exist, skipping.")
            return
            
        for root, dirs, files in os.walk(dir_path):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Skip if no image files in this directory
            if not image_files:
                continue
                
            for img_file in image_files:
                img_path = os.path.join(root, img_file)
                # Check for corresponding text file in the same directory
                txt_path = os.path.join(root, 'text.txt')
                
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                        if caption:  # Only add if caption is not empty
                            self.samples.append((img_path, caption))
                    except Exception as e:
                        logger.warning(f"Error reading text file {txt_path}: {e}")
        
        new_samples = len(self.samples) - start_count
        logger.info(f"Collected {new_samples} samples from {dir_path}")
        
    def _log_dataset_distribution(self):
        """Log distribution of samples across top-level directories"""
        dir_counts = {}
        for img_path, _ in self.samples:
            # Get the first directory after public_data
            parts = img_path.split(os.sep)
            if 'public_data' in parts:
                idx = parts.index('public_data')
                if idx + 1 < len(parts):
                    top_dir = parts[idx + 1]
                    dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1
        
        logger.info("Sample distribution by top-level directory:")
        for dir_name, count in dir_counts.items():
            logger.info(f"  - {dir_name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Process image and text for VQA model
            inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            pixel_values = inputs["pixel_values"].squeeze(0)
            
            # Process target text
            target_encoding = self.processor.tokenizer(caption, padding="max_length", max_length=128, truncation=True)
            target_ids = torch.tensor(target_encoding.input_ids)
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": target_ids,
                "metadata": {
                    "img_path": img_path,
                    "caption": caption
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            # Return a placeholder for failed samples
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


class BLIP2TrainableModel(torch.nn.Module):
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl-coco", device="cuda"):
        super().__init__()
        self.device = device
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Freeze vision encoder for efficiency
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
            
        # Only train the language modeling components
        for param in self.model.language_model.parameters():
            param.requires_grad = True
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        return self.model(
            pixel_values=pixel_values.to(self.device),
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None
        )


def find_subdirectories(base_dir):
    """Find all top-level subdirectories in the base directory"""
    subdirs = []
    # Check if directory exists before attempting to list it
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist")
        return subdirs
        
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    return subdirs


def train_model(config):
    # Add validation for data directory
    if not os.path.exists(config["data_dir"]):
        logger.error(f"Data directory {config['data_dir']} does not exist")
        return None, None
        
    processor = AutoProcessor.from_pretrained(config["model_name"])
    
    # Find all data directories
    all_data_dirs = []
    
    # Add the base data directory if it contains images directly
    base_dir = config["data_dir"]
    image_exists = False
    for root, _, files in os.walk(base_dir):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            image_exists = True
            break
    
    if image_exists:
        all_data_dirs.append(base_dir)
    
    # Find and add all subdirectories
    subdirs = find_subdirectories(base_dir)
    all_data_dirs.extend(subdirs)
    
    logger.info(f"Found {len(all_data_dirs)} potential data directories")
    
    # Create full dataset from all data directories
    full_dataset = ImageCaptioningDataset(
        data_dir=all_data_dirs,
        processor=processor,
        prompt=config["prompt"]
    )
    
    # Check if dataset is large enough
    if len(full_dataset) < 10:
        logger.error(f"Dataset too small with only {len(full_dataset)} samples. Check directory structure.")
        return None, None
    
    # Split dataset into train and validation (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Generate indices for training and validation
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    # Initialize model
    model = BLIP2TrainableModel(config["model_name"], config["device"]).to(config["device"])
    
    # Log model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} total parameters, {trainable_params:,} are trainable ({trainable_params/total_params:.2%})")
    
    # Set up optimizer
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        relative_step=False,
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config["num_epochs"]
    
    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_checkpoint_path = None
    early_stopping_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 3)
    
    # Create a directory for training run with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["output_dir"], f"run_{timestamp}")
    ckpt_dir = config["checkpoint_dir"]
    
    # Create checkpoint directory if it does not exist
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save training samples for reference
    train_samples_file = os.path.join(run_dir, "train_samples.json")
    train_sample_info = []
    for idx in random.sample(train_indices, min(10, len(train_indices))):
        sample = full_dataset.samples[idx]
        train_sample_info.append({
            "image_path": sample[0],
            "caption": sample[1]
        })
    with open(train_samples_file, "w") as f:
        json.dump(train_sample_info, f, indent=2)
    
    # Save val samples for reference
    val_samples_file = os.path.join(run_dir, "val_samples.json")
    val_sample_info = []
    for idx in random.sample(val_indices, min(10, len(val_indices))):
        sample = full_dataset.samples[idx]
        val_sample_info.append({
            "image_path": sample[0],
            "caption": sample[1]
        })
    with open(val_samples_file, "w") as f:
        json.dump(val_sample_info, f, indent=2)
    
    for epoch in range(config["num_epochs"]):
        logger.info(f"Starting epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        model.train()
        total_train_loss = 0
        
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")
        for batch in train_progress_bar:
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(config["device"]),
                input_ids=batch["input_ids"].to(config["device"]),
                attention_mask=batch["attention_mask"].to(config["device"]),
                labels=batch["labels"].to(config["device"])
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]")
            for batch in val_progress_bar:
                outputs = model(
                    pixel_values=batch["pixel_values"].to(config["device"]),
                    input_ids=batch["input_ids"].to(config["device"]),
                    attention_mask=batch["attention_mask"].to(config["device"]),
                    labels=batch["labels"].to(config["device"])
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1} - Average validation loss: {avg_val_loss:.4f}")
        
        # Save model checkpoint for this epoch
        if (epoch + 1) % config.get("save_every", 1) == 0:
            epoch_save_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(epoch_save_path, exist_ok=True)
            model.model.save_pretrained(epoch_save_path)
            processor.save_pretrained(epoch_save_path)
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
        
        # Save best model checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
            
            # Save as best checkpoint
            best_checkpoint_path = os.path.join(ckpt_dir, "best_checkpoint")
            
            # Remove old best checkpoint if it exists
            if os.path.exists(best_checkpoint_path):
                shutil.rmtree(best_checkpoint_path)
                
            # Create new best checkpoint
            os.makedirs(best_checkpoint_path, exist_ok=True)
            model.model.save_pretrained(best_checkpoint_path)
            processor.save_pretrained(best_checkpoint_path)
            
            # Save metadata
            with open(os.path.join(best_checkpoint_path, "metadata.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "validation_loss": avg_val_loss,
                    "training_loss": avg_train_loss,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            logger.info(f"Best model saved to {best_checkpoint_path}")
        else:
            early_stopping_counter += 1
            logger.info(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Create a symlink to the best checkpoint at the top level output directory
    final_best_path = os.path.join(config["output_dir"], "best_checkpoint")
    if os.path.exists(final_best_path):
        if os.path.islink(final_best_path):
            os.unlink(final_best_path)
        else:
            shutil.rmtree(final_best_path)
    
    # Copy the best checkpoint to the top level output directory
    if best_checkpoint_path:
        shutil.copytree(best_checkpoint_path, final_best_path)
        logger.info(f"Copied best checkpoint to {final_best_path}")
    
    return model, final_best_path


def evaluate_model(model_path, processor_path, test_data_dirs, device="cuda", num_samples=None):
    """Evaluate the trained model on test data and generate captions for comparison"""
    # Check if model_path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist")
        return []
    
    # Load model and processor
    logger.info(f"Loading model from {model_path}")
    model = BLIP2TrainableModel(model_path, device).to(device)
    processor = AutoProcessor.from_pretrained(processor_path)
    
    # Create dataset from test directories
    logger.info(f"Creating test dataset from {test_data_dirs}")
    test_dataset = ImageCaptioningDataset(data_dir=test_data_dirs, processor=processor)
    
    # If num_samples is specified, take a random subset
    if num_samples and num_samples < len(test_dataset):
        indices = random.sample(range(len(test_dataset)), num_samples)
        samples = [test_dataset.samples[i] for i in indices]
        logger.info(f"Randomly selected {num_samples} samples for evaluation")
    else:
        samples = test_dataset.samples
        logger.info(f"Using all {len(samples)} samples for evaluation")
        
    results = []
    
    model.eval()
    
    for img_path, ground_truth in tqdm(samples, desc="Evaluating"):
        try:
            # Load image and prepare for inference
            image = Image.open(img_path).convert('RGB')
            prompt = "Look at the image carefully and describe it in one complete sentence."
            
            # Add text parameter and move to device
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # All tensors need to be on the same device
                generated_ids = model.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100
                )
                
            generated_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            
            results.append({
                "image_path": img_path,
                "ground_truth": ground_truth,
                "generated_caption": generated_caption
            })
        except Exception as e:
            logger.error(f"Error evaluating image {img_path}: {e}")
    
    # Calculate success rate
    success_rate = len(results) / len(samples) * 100 if samples else 0
    logger.info(f"Evaluation success rate: {success_rate:.2f}% ({len(results)} out of {len(samples)})")
    
    # Save results to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(model_path), f"evaluation_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")
    return results


if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()
    
    config = {
        "model_name": args.model_name,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "prompt": args.prompt,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "max_grad_norm": 1.0,
        "early_stopping_patience": args.early_stopping_patience,
        "save_every": args.save_every,
        "checkpoint_dir": args.checkpoint_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Starting training using ALL available image-caption pairs...")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    trained_model, best_checkpoint_path = train_model(config)
    
    if best_checkpoint_path:
        logger.info(f"Training complete. Best model saved at: {best_checkpoint_path}")
        logger.info("Starting evaluation using the best checkpoint...")
        
        # Use the same data directories for evaluation
        test_data_dirs = config["data_dir"]
        
        evaluation_results = evaluate_model(
            model_path=best_checkpoint_path,
            processor_path=best_checkpoint_path,
            test_data_dirs=test_data_dirs,
            device=config["device"],
            num_samples=args.eval_samples
        )
        
        logger.info("Evaluation complete!")
        
        # Print some sample results
        for i in range(min(5, len(evaluation_results))):
            result = evaluation_results[i]
            logger.info(f"\nImage: {result['image_path']}")
            logger.info(f"Ground Truth: {result['ground_truth']}")
            logger.info(f"Generated: {result['generated_caption']}")
    else:
        logger.error("Training failed. No model was produced.")