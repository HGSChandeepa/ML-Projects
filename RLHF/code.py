# Complete RLHF Implementation for LLM Fine-tuning
# Designed for Google Colab - Ready to Run!

# ============================================================================
# INSTALLATION AND SETUP
# ============================================================================

# First, install required packages
!pip install transformers datasets trl peft accelerate bitsandbytes wandb torch evaluate
!pip install --upgrade transformers[torch]

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, BitsAndBytesConfig
)
from datasets import Dataset, load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CONFIGURATION
# ============================================================================

class RLHFConfig:
    """Configuration class for RLHF training"""
    
    # Model settings
    BASE_MODEL = "microsoft/DialoGPT-small"  # Smaller model for Colab
    TOKENIZER_MODEL = BASE_MODEL
    
    # LoRA settings for efficient training
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    # Training hyperparameters
    SFT_EPOCHS = 1
    REWARD_MODEL_EPOCHS = 1
    PPO_STEPS = 50
    
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 256
    
    # PPO specific
    PPO_BATCH_SIZE = 2
    PPO_MINI_BATCH_SIZE = 1
    
    # Paths
    OUTPUT_DIR = "/content/rlhf_output"
    SFT_MODEL_PATH = "/content/sft_model"
    REWARD_MODEL_PATH = "/content/reward_model"

config = RLHFConfig()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.SFT_MODEL_PATH, exist_ok=True)
os.makedirs(config.REWARD_MODEL_PATH, exist_ok=True)

# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_sample_datasets():
    """
    Create sample datasets for demonstration.
    In practice, you'd load your own datasets.
    """
    
    # Sample SFT dataset (prompt-response pairs)
    sft_data = [
        {"prompt": "How do I cook pasta?", 
         "response": "Boil water, add salt, add pasta, cook for 8-10 minutes, drain, and serve."},
        {"prompt": "What is machine learning?", 
         "response": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."},
        {"prompt": "How do I stay healthy?", 
         "response": "Eat balanced meals, exercise regularly, get enough sleep, and manage stress effectively."},
        {"prompt": "Explain photosynthesis", 
         "response": "Photosynthesis is the process where plants convert sunlight, water, and CO2 into glucose and oxygen."},
        {"prompt": "What is Python?", 
         "response": "Python is a high-level, interpreted programming language known for its simplicity and versatility."},
    ] * 20  # Repeat for more training data
    
    # Sample preference dataset (prompt, chosen, rejected)
    preference_data = [
        {
            "prompt": "How do I learn programming?",
            "chosen": "Start with basics like variables and loops, practice coding daily, and build projects to apply your knowledge.",
            "rejected": "Just copy code from the internet and you'll figure it out eventually."
        },
        {
            "prompt": "What's the best way to study?",
            "chosen": "Create a study schedule, use active learning techniques, take breaks, and test yourself regularly.",
            "rejected": "Just read everything once and hope you remember it during the exam."
        },
        {
            "prompt": "How do I stay motivated?",
            "chosen": "Set clear goals, track progress, celebrate small wins, and surround yourself with supportive people.",
            "rejected": "Motivation doesn't matter, just force yourself to do things even if you hate them."
        },
    ] * 15  # Repeat for more training data
    
    return sft_data, preference_data

# Create datasets
sft_data, preference_data = create_sample_datasets()
print(f"Created {len(sft_data)} SFT examples and {len(preference_data)} preference examples")

# ============================================================================
# TOKENIZER AND MODEL UTILITIES
# ============================================================================

def setup_tokenizer_and_model(model_name: str, for_reward_model: bool = False):
    """Setup tokenizer and model with proper configurations"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    if for_reward_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # For reward scoring
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    return tokenizer, model

# ============================================================================
# STEP 1: SUPERVISED FINE-TUNING (SFT)
# ============================================================================

def prepare_sft_dataset(data: List[Dict], tokenizer, max_length: int = 256):
    """Prepare dataset for supervised fine-tuning"""
    
    def tokenize_function(examples):
        # Combine prompt and response
        texts = [f"Human: {prompt}\nAssistant: {response}{tokenizer.eos_token}" 
                for prompt, response in zip(examples['prompt'], examples['response'])]
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Convert to HF dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def train_sft_model():
    """Train the base model using supervised fine-tuning"""
    
    print("🚀 Starting Supervised Fine-tuning (SFT)...")
    
    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(config.BASE_MODEL)
    
    # Setup LoRA for efficient training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"]  # For DialoGPT
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset = prepare_sft_dataset(sft_data, tokenizer, config.MAX_LENGTH)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.SFT_MODEL_PATH,
        num_train_epochs=config.SFT_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,  # Use mixed precision
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Training SFT model...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(config.SFT_MODEL_PATH)
    
    print(f"✅ SFT model saved to {config.SFT_MODEL_PATH}")
    
    return model, tokenizer

# ============================================================================
# STEP 2: REWARD MODEL TRAINING
# ============================================================================

def prepare_preference_dataset(data: List[Dict], tokenizer, max_length: int = 256):
    """Prepare dataset for reward model training"""
    
    processed_data = []
    
    for item in data:
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Create chosen example (label = 1)
        chosen_text = f"Human: {prompt}\nAssistant: {chosen}"
        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        
        processed_data.append({
            'input_ids': chosen_tokens['input_ids'][0],
            'attention_mask': chosen_tokens['attention_mask'][0],
            'labels': 1.0  # Higher reward for chosen
        })
        
        # Create rejected example (label = 0)
        rejected_text = f"Human: {prompt}\nAssistant: {rejected}"
        rejected_tokens = tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        
        processed_data.append({
            'input_ids': rejected_tokens['input_ids'][0],
            'attention_mask': rejected_tokens['attention_mask'][0],
            'labels': 0.0  # Lower reward for rejected
        })
    
    return Dataset.from_list(processed_data)

def train_reward_model():
    """Train a reward model to score model outputs"""
    
    print("🎯 Starting Reward Model Training...")
    
    # Setup tokenizer and reward model
    tokenizer, reward_model = setup_tokenizer_and_model(
        config.BASE_MODEL, 
        for_reward_model=True
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"]
    )
    
    reward_model = get_peft_model(reward_model, lora_config)
    
    # Prepare dataset
    reward_dataset = prepare_preference_dataset(
        preference_data, 
        tokenizer, 
        config.MAX_LENGTH
    )
    
    # Split into train/validation
    train_size = int(0.8 * len(reward_dataset))
    train_dataset = reward_dataset.select(range(train_size))
    eval_dataset = reward_dataset.select(range(train_size, len(reward_dataset)))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.REWARD_MODEL_PATH,
        num_train_epochs=config.REWARD_MODEL_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=25,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
    )
    
    # Custom trainer for reward model
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss = nn.MSELoss()(logits, labels)
            return (loss, outputs) if return_outputs else loss
    
    # Initialize trainer
    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("Training reward model...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(config.REWARD_MODEL_PATH)
    
    print(f"✅ Reward model saved to {config.REWARD_MODEL_PATH}")
    
    return reward_model, tokenizer

# ============================================================================
# STEP 3: PPO TRAINING
# ============================================================================

def prepare_ppo_dataset(prompts: List[str], tokenizer, max_length: int = 128):
    """Prepare prompts for PPO training"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({"prompt": prompts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def train_with_ppo():
    """Fine-tune the SFT model using PPO with reward model feedback"""
    
    print("🔄 Starting PPO Training...")
    
    # Load SFT model and tokenizer
    sft_tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_PATH)
    if sft_tokenizer.pad_token is None:
        sft_tokenizer.pad_token = sft_tokenizer.eos_token
    
    # Load SFT model for PPO (with value head)
    sft_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.SFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load reward model
    reward_tokenizer = AutoTokenizer.from_pretrained(config.REWARD_MODEL_PATH)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.REWARD_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # PPO configuration
    ppo_config = PPOConfig(
        model_name=config.BASE_MODEL,
        learning_rate=1e-5,
        batch_size=config.PPO_BATCH_SIZE,
        mini_batch_size=config.PPO_MINI_BATCH_SIZE,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        ppo_epochs=1,
        seed=42,
    )
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=sft_model,
        tokenizer=sft_tokenizer,
        dataset=None,  # We'll provide data manually
    )
    
    # Prepare prompts for PPO
    ppo_prompts = [item['prompt'] for item in sft_data[:20]]  # Use subset
    ppo_dataset = prepare_ppo_dataset(ppo_prompts, sft_tokenizer, 128)
    
    # Training loop
    for epoch in range(config.PPO_STEPS):
        print(f"PPO Step {epoch + 1}/{config.PPO_STEPS}")
        
        # Sample batch
        batch = ppo_dataset.shuffle().select(range(min(config.PPO_BATCH_SIZE, len(ppo_dataset))))
        query_tensors = [torch.tensor(item['input_ids']) for item in batch]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_length=config.MAX_LENGTH,
            do_sample=True,
            temperature=0.7,
            pad_token_id=sft_tokenizer.pad_token_id
        )
        
        # Compute rewards
        rewards = []
        for query_tensor, response_tensor in zip(query_tensors, response_tensors):
            # Decode the full conversation
            full_text = sft_tokenizer.decode(response_tensor, skip_special_tokens=True)
            
            # Tokenize for reward model
            reward_inputs = reward_tokenizer(
                full_text,
                truncation=True,
                padding=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt"
            ).to(reward_model.device)
            
            # Get reward score
            with torch.no_grad():
                reward_output = reward_model(**reward_inputs)
                reward = reward_output.logits.squeeze().item()
            
            rewards.append(torch.tensor(reward))
        
        # Update model with PPO
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log statistics
        if epoch % 10 == 0:
            print(f"Mean reward: {np.mean([r.item() for r in rewards]):.3f}")
    
    # Save the final model
    ppo_output_dir = os.path.join(config.OUTPUT_DIR, "ppo_model")
    ppo_trainer.save_pretrained(ppo_output_dir)
    sft_tokenizer.save_pretrained(ppo_output_dir)
    
    print(f"✅ PPO model saved to {ppo_output_dir}")
    
    return sft_model, sft_tokenizer

# ============================================================================
# EVALUATION AND TESTING
# ============================================================================

def test_models():
    """Test and compare the models at different stages"""
    
    print("🧪 Testing Models...")
    
    test_prompts = [
        "How do I learn to cook?",
        "What's the best way to exercise?",
        "How can I improve my communication skills?",
    ]
    
    # Test original model
    print("\n--- Original Model ---")
    tokenizer, original_model = setup_tokenizer_and_model(config.BASE_MODEL)
    original_pipeline = pipeline(
        "text-generation",
        model=original_model,
        tokenizer=tokenizer,
        max_length=150,
        do_sample=True,
        temperature=0.7
    )
    
    for prompt in test_prompts:
        response = original_pipeline(f"Human: {prompt}\nAssistant:")
        print(f"Q: {prompt}")
        print(f"A: {response[0]['generated_text'].split('Assistant:')[-1].strip()}\n")
    
    # Test SFT model if available
    if os.path.exists(config.SFT_MODEL_PATH):
        print("\n--- SFT Model ---")
        sft_tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_PATH)
        sft_model = AutoModelForCausalLM.from_pretrained(
            config.SFT_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        sft_pipeline = pipeline(
            "text-generation",
            model=sft_model,
            tokenizer=sft_tokenizer,
            max_length=150,
            do_sample=True,
            temperature=0.7
        )
        
        for prompt in test_prompts:
            response = sft_pipeline(f"Human: {prompt}\nAssistant:")
            print(f"Q: {prompt}")
            print(f"A: {response[0]['generated_text'].split('Assistant:')[-1].strip()}\n")
    
    # Test final PPO model if available
    ppo_model_path = os.path.join(config.OUTPUT_DIR, "ppo_model")
    if os.path.exists(ppo_model_path):
        print("\n--- PPO Model ---")
        ppo_tokenizer = AutoTokenizer.from_pretrained(ppo_model_path)
        ppo_model = AutoModelForCausalLM.from_pretrained(
            ppo_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        ppo_pipeline = pipeline(
            "text-generation",
            model=ppo_model,
            tokenizer=ppo_tokenizer,
            max_length=150,
            do_sample=True,
            temperature=0.7
        )
        
        for prompt in test_prompts:
            response = ppo_pipeline(f"Human: {prompt}\nAssistant:")
            print(f"Q: {prompt}")
            print(f"A: {response[0]['generated_text'].split('Assistant:')[-1].strip()}\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_full_rlhf_pipeline():
    """Run the complete RLHF pipeline"""
    
    print("🚀 Starting Full RLHF Pipeline!")
    print("=" * 60)
    
    try:
        # Step 1: Supervised Fine-tuning
        print("\nSTEP 1: Supervised Fine-tuning")
        sft_model, sft_tokenizer = train_sft_model()
        
        # Step 2: Reward Model Training
        print("\nSTEP 2: Reward Model Training")
        reward_model, reward_tokenizer = train_reward_model()
        
        # Step 3: PPO Training
        print("\nSTEP 3: PPO Training")
        ppo_model, ppo_tokenizer = train_with_ppo()
        
        # Step 4: Evaluation
        print("\nSTEP 4: Model Evaluation")
        test_models()
        
        print("\n🎉 RLHF Pipeline completed successfully!")
        print(f"Final model saved in: {config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error in RLHF pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# QUICK START FUNCTIONS
# ============================================================================

def quick_demo():
    """Run a quick demo with minimal training for testing"""
    
    print("🚀 Running Quick RLHF Demo...")
    
    # Reduce training parameters for quick demo
    config.SFT_EPOCHS = 1
    config.REWARD_MODEL_EPOCHS = 1
    config.PPO_STEPS = 10
    config.BATCH_SIZE = 2
    
    # Run pipeline
    run_full_rlhf_pipeline()

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

print("""
🤖 RLHF Implementation Ready!

USAGE OPTIONS:

1. Quick Demo (recommended for first run):
   quick_demo()

2. Full Pipeline:
   run_full_rlhf_pipeline()

3. Individual Steps:
   - train_sft_model()           # Step 1: Supervised fine-tuning
   - train_reward_model()        # Step 2: Reward model training
   - train_with_ppo()           # Step 3: PPO optimization
   - test_models()              # Evaluate models

4. Test existing models:
   test_models()

NOTES:
- This implementation uses LoRA for memory efficiency
- Designed to work within Google Colab's resource constraints
- Uses sample data - replace with your own datasets for real applications
- Monitor GPU memory usage during training

Ready to start? Run: quick_demo()
""")

# Uncomment the next line to run automatically:
# quick_demo()