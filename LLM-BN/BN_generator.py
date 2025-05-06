import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
from typing import List, Dict
import re
from enum import Enum
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

class TabularDataset(Dataset):
    """Dataset for training GPT-2"""
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, 
            truncation=True,
            max_length=1024,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class BNGenerator:
    """Main LLM-BN model class"""
    def __init__(self, model_name='distilgpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.optimizer = None
    
    def init_optimizer(self, learning_rate=5e-5):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
    def prepare_data(self, texts: List[str], batch_size: int = 8) -> DataLoader:
        """Prepare data for training"""
        dataset = TabularDataset(texts, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        loss = outputs.loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, 
              dataloader: DataLoader,
              epochs: int = 10,
              learning_rate: float = 5e-5):
        """Train the model"""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                batch_loss = self.train_step(batch)
                total_loss += batch_loss
                
                progress_bar.set_postfix({
                    'batch': f'{batch_idx+1}/{len(dataloader)}',
                    'loss': f'{batch_loss:.4f}'
                })
            
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def generate(self, n_samples: int, condition: str = None, max_length: int = 200, 
                 temperature: float = 0.7, batch_size: int = 100) -> List[str]:
        """Generate new samples with true batch processing"""
        self.model.eval()
        generated_texts = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            with tqdm(total=n_samples, desc="Total Progress") as pbar:
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)
                    current_batch_size = end_idx - start_idx
                    
                    pbar.set_description(f"Total Progress (Batch {batch_idx+1}/{n_batches})")
                    
                    # Process the entire batch at once
                    if condition:
                        input_text = [condition] * current_batch_size
                    else:
                        input_text = ["Education stauts is"] * current_batch_size
                        
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        num_return_sequences=1,
                        use_cache=True
                    )
                    
                    batch_texts = [
                        self.tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    
                    generated_texts.extend(batch_texts)
                    pbar.update(current_batch_size)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        return generated_texts

    def save_model(self, save_path: str):
        """Save the trained model and tokenizer"""
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(model_dir / "model")
        self.tokenizer.save_pretrained(model_dir / "tokenizer")
        print(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load a trained model and tokenizer"""
        model_dir = Path(load_path)
        
        self.model = GPT2LMHeadModel.from_pretrained(model_dir / "model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir / "tokenizer")
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

class TextDecoder:
    """Decode generated text back to tabular format."""
    def __init__(self):
        
        self.patterns = {
            'Age': r'Age group is (\[\d+,\d+\)|\[85,90\])', 
            'Gender': r'Gender is (Male|Female)',
            'Homeincome': r'Household monthly income level is (< 1M KRW|1M-3M KRW|3M-5M KRW|5M-10M KRW|> 10M KRW)',
            'Hometype': r'Home type is (Apartment|Villa|Multi-family|Single-family|Studio-type residence|Other)',
            'CarOwn': r'Car ownership of household is (Yes|No)', 
            'Driver': r'Driver license is (Yes|No)',
            'Workdays': r'Work days is (5 days|6 days|1~4 days|Inoccupation/non-regular)',
            'Worktype': r'Work type is (Student|Inoccupation/Housewife|Experts|Service|Sales|Manager/Office|Agriculture/fisher|Simple labor|Other)',
            'Student': r'Education status is (Elementary/Middle/High School|Preschool|University|Not student)',
            'NumHH': r'Number of household members is (\d+)',
            'KidinHH': r'Kid in household is (Yes|No)',
            'ComMode': r'Major travel mode is (Car|Public Transportation|Walking|Bike/Bicycle|Taxi|No commute)',
            'ComTime': r'Major departure time is (Peak|NonPeak|Others|No commute)'
        }

        self.gender_map = {'Male': 1, 'Female': 2}
        self.binary_map = {'Yes': 1, 'No': 2}
        self.homeincome_map = {
            '< 1M KRW': 1,
            '1M-3M KRW': 2,
            '3M-5M KRW': 3,
            '5M-10M KRW': 4,
            '> 10M KRW': 5
        }
        self.hometype_map = {
            'Apartment': 1,
            'Villa': 2,
            'Multi-family': 3,
            'Single-family': 4,
            'Studio-type residence': 5,
            'Other': 6
        }
        self.workdays_map = {
            '5 days': 1,
            '6 days': 2,
            '1~4 days': 3,
            'Inoccupation/non-regular': 4
        }
        self.worktype_map = {
            'Student': 1,
            'Inoccupation/Housewife': 2,
            'Experts': 3,
            'Service': 4,
            'Sales': 5,
            'Manager/Office': 6,
            'Agriculture/fisher': 7,
            'Simple labor': 8,
            'Other': 9
        }
        self.student_map = {
            'Elementary/Middle/High School': 1,
            'Preschool': 2,
            'University': 3,
            'Not student': 4
        }

    def decode_text(self, text: str) -> Dict:
        result = {}
    
        for col, pattern in self.patterns.items():
            match = re.search(pattern, text)
    
            if match:
                value = match.group(1).strip()
     
                if col == 'Age':
                    if re.match(r'^(?:\[\d+,\d+\)|\[85,90\])$', value):
                        result[col] = value
                    else:
                        result[col] = np.nan

                elif col == 'Gender':
                    result[col] = self.gender_map.get(value, np.nan)
    
                elif col == 'Homeincome':
                    result[col] = self.homeincome_map.get(value, np.nan)
    
                elif col == 'Hometype':
                    result[col] = self.hometype_map.get(value, np.nan)
    
                elif col in ['CarOwn', 'Driver', 'KidinHH']:
                    result[col] = self.binary_map.get(value, np.nan)
    
                elif col == 'Workdays':
                    result[col] = self.workdays_map.get(value, np.nan)
    
                elif col == 'Worktype':
                    result[col] = self.worktype_map.get(value, np.nan)
    
                elif col == 'Student':
                    result[col] = self.student_map.get(value, np.nan)
    
                elif col == 'NumHH':
                    try:
                        val = int(value)
                        result[col] = val if 1 <= val <= 7 else np.nan
                    except (ValueError, TypeError):
                        result[col] = np.nan
    
                elif col in ['ComMode', 'ComTime']:
                    result[col] = value
    
            else:
                result[col] = np.nan
    
        int_columns = ['Gender', 'Homeincome', 'Hometype', 'CarOwn', 'Driver',
                       'Workdays', 'Worktype', 'Student', 'NumHH', 'KidinHH']
        for col in int_columns:
            if col in result and pd.notna(result[col]):
                result[col] = int(result[col])
                
        return result


    def decode_texts(self, texts: List[str]) -> pd.DataFrame:
        decoded = [self.decode_text(t) for t in texts]
        df = pd.DataFrame(decoded)
        
        int_columns = ['Gender', 'Homeincome', 'Hometype', 'CarOwn', 'Driver',
                      'Workdays', 'Worktype', 'Student', 'NumHH', 'KidinHH']
        
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype('Int64')
                
        return df