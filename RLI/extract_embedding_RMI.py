import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
from transformers import  AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

def parse_config():
    parser = argparse.ArgumentParser(description="Compute embeddings for RNA and drugs")
    
    # Data related parameters
    parser.add_argument("--csv_path", type=str, required=True,
                      help="Input CSV file path containing RNA and drug sequences")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for saving embeddings and mask files")
    

    parser.add_argument("--prot_id_col", type=str, default="RNA_ID", help="RNA ID column name")
    parser.add_argument("--prot_seq_col", type=str, default="RNA_Sequence", help="RNA sequence column name")
    parser.add_argument("--drug_id_col", type=str, default="Drug_ID", help="Drug ID column name")
    parser.add_argument("--drug_seq_col", type=str, default="Drug_Sequence", help="Drug sequence column name")
    parser.add_argument("--label_col", type=str, default="label", help="Label column name")
    
    # Model related parameters
    parser.add_argument("--prot_encoder_path", type=str, required=True,
                      help="RNA encoder model path")
    parser.add_argument("--drug_encoder_path", type=str, default="ibm/MoLFormer-XL-both-10pct",
                      help="Drug encoder model path")
    parser.add_argument("--prot_max_length", type=int, default=128,
                      help="Maximum length of RNA sequences")
    parser.add_argument("--drug_max_length", type=int, default=128,
                      help="Maximum length of drug sequences")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Number of sequences to process per batch")
    
    return parser.parse_args()

# Function to load models and tokenizers
def load_models(prot_encoder_path, drug_encoder_path, device):


    config = BertConfig.from_pretrained(prot_encoder_path)
    prot_tokenizer = AutoTokenizer.from_pretrained(prot_encoder_path, trust_remote_code=True)  
    prot_model = AutoModel.from_pretrained(prot_encoder_path, trust_remote_code=True, config=config)       
    prot_model = prot_model.to(device) 
    

    drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder_path, trust_remote_code=True)
    drug_model = AutoModel.from_pretrained(drug_encoder_path, deterministic_eval=True, trust_remote_code=True)
    drug_model = drug_model.to(device)
    
    return prot_model, prot_tokenizer, drug_model, drug_tokenizer

class PreEncoded:
    def __init__(self, prot_model, drug_model, device):
        self.prot_encoder = prot_model
        self.drug_encoder = drug_model
        self.device = device
    
    def encoding(self, prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask):
        with torch.no_grad():
            # Compute RNA embeddings
            prot_embed = self.prot_encoder(
                input_ids=prot_input_ids, 
                attention_mask=prot_attention_mask, 
                return_dict=True
            )[0]     
            
            # Compute drug embeddings
            drug_embed = self.drug_encoder(
                input_ids=drug_input_ids, 
                attention_mask=drug_attention_mask, 
                return_dict=True
            ).last_hidden_state
            
       
        prot_embed = prot_embed.cpu().detach().numpy()
        drug_embed = drug_embed.cpu().detach().numpy()
        
      
        if len(prot_embed.shape) == 2:
            prot_embed = prot_embed.reshape(1, *prot_embed.shape)
        if len(drug_embed.shape) == 2:
            drug_embed = drug_embed.reshape(1, *drug_embed.shape)
            
        return prot_embed, drug_embed


def process_csv_and_compute_embeddings(csv_path, output_dir, prot_encoder_path, drug_encoder_path, 
                                      prot_max_length=1024, drug_max_length=1024, batch_size=32,
                                      prot_id_col="RNA_ID", prot_seq_col="RNA_Sequence", drug_id_col="Drug_ID", drug_seq_col="Drug_Sequence", label_col="label"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
 
    prot_model, prot_tokenizer, drug_model, drug_tokenizer = load_models(
        prot_encoder_path, drug_encoder_path, device)
    

    encoder = PreEncoded(prot_model, drug_model, device)
    

    df = pd.read_csv(csv_path)
    print(f"CSV file loaded successfully with {len(df)} records")
    
  
    # Check if column names exist
    for col in [prot_id_col, prot_seq_col, drug_id_col, drug_seq_col]:
        if col not in df.columns:
            raise ValueError(f"Column name {col} not found in CSV file, please check parameters or file format!")
    
    # Extract unique RNAs and drugs
    unique_prots = df[[prot_id_col, prot_seq_col]].drop_duplicates()
    unique_drugs = df[[drug_id_col, drug_seq_col]].drop_duplicates()
    
    # Ensure we have valid DataFrames
    if unique_prots is not None and len(unique_prots) > 0:
        print(f"Extracted {len(unique_prots)} unique RNA sequences")
    else:
        print("No unique RNA sequences found")
        unique_prots = pd.DataFrame(columns=[prot_id_col, prot_seq_col])
        
    if unique_drugs is not None and len(unique_drugs) > 0:
        print(f"Extracted {len(unique_drugs)} unique drug sequences")
    else:
        print("No unique drug sequences found")
        unique_drugs = pd.DataFrame(columns=[drug_id_col, drug_seq_col])
    
   
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process RNA sequences in batches
    print("Processing RNA sequences in batches...")
 
    if isinstance(unique_prots, pd.DataFrame):
        prot_batches = [unique_prots.iloc[i:i+batch_size] for i in range(0, len(unique_prots), batch_size)]
    else:
        prot_batches = []
    for batch_idx, batch in enumerate(tqdm(prot_batches)):
        prot_embeddings_batch = {}
        prot_masks_batch = {}
        
       
        if isinstance(batch, pd.DataFrame):
            for _, row in batch.iterrows():
                prot_id = row[prot_id_col]
                prot_seq = row[prot_seq_col]
                
               
                prot_encoding = prot_tokenizer.batch_encode_plus(
                    [prot_seq],
                    max_length=prot_max_length,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                prot_input_ids = prot_encoding["input_ids"].to(device)
                prot_attention_mask = prot_encoding["attention_mask"].to(device)
                
        
                prot_masks_batch[prot_id] = prot_attention_mask.cpu().numpy()
                
                dummy_drug_input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)
                dummy_drug_attention_mask = torch.zeros((1, 1), dtype=torch.long).to(device)
                
       
                prot_embed, _ = encoder.encoding(
                    prot_input_ids, prot_attention_mask, 
                    dummy_drug_input_ids, dummy_drug_attention_mask
                )
                
         
                if len(prot_embed.shape) != 3:
                    prot_embed = prot_embed.reshape(1, *prot_embed.shape)
                

                prot_embeddings_batch[prot_id] = prot_embed
                
       
                torch.cuda.empty_cache()
        
      
        with open(os.path.join(temp_dir, f"protein_embeddings_batch_{batch_idx}.pkl"), "wb") as f:
            pickle.dump(prot_embeddings_batch, f)
        with open(os.path.join(temp_dir, f"protein_masks_batch_{batch_idx}.pkl"), "wb") as f:
            pickle.dump(prot_masks_batch, f)
    
   
    print("Processing drug sequences in batches...")
    if isinstance(unique_drugs, pd.DataFrame):
        drug_batches = [unique_drugs.iloc[i:i+batch_size] for i in range(0, len(unique_drugs), batch_size)]
    else:
        drug_batches = []
    for batch_idx, batch in enumerate(tqdm(drug_batches)):
        drug_embeddings_batch = {}
        drug_masks_batch = {}
        
       
        if isinstance(batch, pd.DataFrame):
            for _, row in batch.iterrows():
                drug_id = row[drug_id_col]
                drug_seq = row[drug_seq_col]
                
                # Tokenize
                drug_encoding = drug_tokenizer.batch_encode_plus(
                    [drug_seq],
                    max_length=drug_max_length,
                    padding="max_length",
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                
                drug_input_ids = drug_encoding["input_ids"].to(device)
                drug_attention_mask = drug_encoding["attention_mask"].to(device)
                
          
                drug_masks_batch[drug_id] = drug_attention_mask.cpu().numpy()
                
      
                dummy_prot_input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)
                dummy_prot_attention_mask = torch.zeros((1, 1), dtype=torch.long).to(device)
                
            
                _, drug_embed = encoder.encoding(
                    dummy_prot_input_ids, dummy_prot_attention_mask,
                    drug_input_ids, drug_attention_mask
                )
                
              
                if len(drug_embed.shape) != 3:
                    drug_embed = drug_embed.reshape(1, *drug_embed.shape)
                
          
                drug_embeddings_batch[drug_id] = drug_embed
                
           
                torch.cuda.empty_cache()
        
   
        with open(os.path.join(temp_dir, f"drug_embeddings_batch_{batch_idx}.pkl"), "wb") as f:
            pickle.dump(drug_embeddings_batch, f)
        with open(os.path.join(temp_dir, f"drug_masks_batch_{batch_idx}.pkl"), "wb") as f:
            pickle.dump(drug_masks_batch, f)
    

    print("Merging results from all batches...")
    prot_embeddings = {}
    prot_masks = {}
    drug_embeddings = {}
    drug_masks = {}
    
  
    for batch_file in os.listdir(temp_dir):
        if batch_file.startswith("protein_embeddings_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
               
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                prot_embeddings.update(batch_data)
        elif batch_file.startswith("protein_masks_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
              
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                prot_masks.update(batch_data)
    

    for batch_file in os.listdir(temp_dir):
        if batch_file.startswith("drug_embeddings_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
              
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                drug_embeddings.update(batch_data)
        elif batch_file.startswith("drug_masks_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
        
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                drug_masks.update(batch_data)
    
  
    print("Saving final results...")
 
    prot_embeddings_np = {k: v.numpy() for k, v in prot_embeddings.items()}
    prot_masks_np = {k: v.numpy() for k, v in prot_masks.items()}
    drug_embeddings_np = {k: v.numpy() for k, v in drug_embeddings.items()}
    drug_masks_np = {k: v.numpy() for k, v in drug_masks.items()}
    
    with open(os.path.join(output_dir, "protein_embeddings.pkl"), "wb") as f:
        pickle.dump(prot_embeddings_np, f)
    with open(os.path.join(output_dir, "protein_masks.pkl"), "wb") as f:
        pickle.dump(prot_masks_np, f)
    with open(os.path.join(output_dir, "drug_embeddings.pkl"), "wb") as f:
        pickle.dump(drug_embeddings_np, f)
    with open(os.path.join(output_dir, "drug_masks.pkl"), "wb") as f:
        pickle.dump(drug_masks_np, f)
    
   
    print("Cleaning up temporary files...")
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    print(f"Embeddings and masks computation completed! Results saved to {output_dir}")
    return prot_embeddings_np, prot_masks_np, drug_embeddings_np, drug_masks_np


if __name__ == "__main__":

    args = parse_config()
    
   
    prot_embeddings, prot_masks, drug_embeddings, drug_masks = process_csv_and_compute_embeddings(
        args.csv_path, 
        args.output_dir, 
        args.prot_encoder_path, 
        args.drug_encoder_path,
        args.prot_max_length,
        args.drug_max_length,
        args.batch_size,
        args.prot_id_col,
        args.prot_seq_col,
        args.drug_id_col,
        args.drug_seq_col,
        args.label_col
    )