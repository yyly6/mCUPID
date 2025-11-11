import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import multimolecule
from transformers import  AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

def parse_config():
    parser = argparse.ArgumentParser(description="Compute embeddings for miRNA and mRNA")
    
    # Data related parameters
    parser.add_argument("--csv_path", type=str, required=True,
                      help="Input CSV file path containing miRNA and mRNA sequences")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for saving embeddings and mask files")
    
    # Model related parameters
    parser.add_argument("--prot_encoder_path", type=str, required=True,
                      help="miRNA encoder model path")
    parser.add_argument("--drug_encoder_path", type=str, required=True,
                      help="mRNA encoder model path")
    parser.add_argument("--prot_max_length", type=int, default=64,
                      help="Maximum length of miRNA sequences")
    parser.add_argument("--drug_max_length", type=int, default=64,
                      help="Maximum length of mRNA sequences")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="Number of sequences to process per batch")
    
    return parser.parse_args()

# Function to load models and tokenizers
def load_models(prot_encoder_path, drug_encoder_path, device):
    # Load miRNA model and tokenizer
    prot_tokenizer = AutoTokenizer.from_pretrained(args.prot_encoder_path)
    prot_model = AutoModel.from_pretrained(args.prot_encoder_path)
    prot_model = prot_model.to(device)
    
    # Load mRNA model and tokenizer
    config = BertConfig.from_pretrained(drug_encoder_path)
    drug_tokenizer = AutoTokenizer.from_pretrained(drug_encoder_path, trust_remote_code=True)
    drug_model = AutoModel.from_pretrained(drug_encoder_path, trust_remote_code=True, config=config)
    drug_model = drug_model.to(device)
    
    return prot_model, prot_tokenizer, drug_model, drug_tokenizer


class PreEncoded:
    def __init__(self, prot_model, drug_model, device):
        self.prot_encoder = prot_model
        self.drug_encoder = drug_model
        self.device = device
    
    def encoding(self, prot_input_ids, prot_attention_mask, drug_input_ids, drug_attention_mask):
        with torch.no_grad():
            # Compute miRNA embeddings
            prot_embed = self.prot_encoder(
                input_ids=prot_input_ids, 
                attention_mask=prot_attention_mask, 
                return_dict=True
            ).last_hidden_state
            
            # Compute mRNA embeddings
            drug_embed = self.drug_encoder(
                input_ids=drug_input_ids, 
                attention_mask=drug_attention_mask, 
                return_dict=True
            )[0]
            

        prot_embed = prot_embed.cpu().detach().numpy()
        drug_embed = drug_embed.cpu().detach().numpy()
        

        if len(prot_embed.shape) == 2:
            prot_embed = prot_embed.reshape(1, *prot_embed.shape)
        if len(drug_embed.shape) == 2:
            drug_embed = drug_embed.reshape(1, *drug_embed.shape)
            
        return prot_embed, drug_embed

def process_csv_and_compute_embeddings(csv_path, output_dir, prot_encoder_path, drug_encoder_path, 
                                      prot_max_length=1024, drug_max_length=1024, batch_size=32):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    

    prot_model, prot_tokenizer, drug_model, drug_tokenizer = load_models(
        prot_encoder_path, drug_encoder_path, device)
    

    encoder = PreEncoded(prot_model, drug_model, device)


    df = pd.read_csv(csv_path)
    print(f"CSV file loaded successfully, {len(df)} records in total")
    
    # Ensure column names exist
    required_columns = ['miRNA_ID', 'mRNA_ID', 'miRNA_Sequence', 'mRNA_Sequence']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file is missing required column: {col}")

    prot_id_col = 'miRNA_ID'  
    prot_seq_col = 'miRNA_Sequence'  
    drug_id_col = 'mRNA_ID' 
    drug_seq_col = 'mRNA_Sequence'  

    # Extract unique miRNA and mRNA
    unique_prots = df[[prot_id_col, prot_seq_col]].drop_duplicates()
    unique_drugs = df[[drug_id_col, drug_seq_col]].drop_duplicates()

    if unique_prots is None:
        unique_prots = df[[prot_id_col, prot_seq_col]].copy()
    if unique_drugs is None:
        unique_drugs = df[[drug_id_col, drug_seq_col]].copy()


    if isinstance(unique_prots, pd.DataFrame):
        unique_prots = unique_prots[unique_prots[prot_seq_col].notna() & (unique_prots[prot_seq_col].str.strip() != '')]
    if isinstance(unique_drugs, pd.DataFrame):
        unique_drugs = unique_drugs[unique_drugs[drug_seq_col].notna() & (unique_drugs[drug_seq_col].str.strip() != '')]

    print(f"Extracted {len(unique_prots)} unique miRNA sequences")
    print(f"Extracted {len(unique_drugs)} unique mRNA sequences")
    

    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Processing miRNA sequences in batches...")
    prot_batches = [unique_prots.iloc[i:i+batch_size] for i in range(0, len(unique_prots), batch_size)]
    for batch_idx, batch in enumerate(tqdm(prot_batches)):
   
        if not isinstance(batch, pd.DataFrame):
            batch = pd.DataFrame(batch)
        prot_embeddings_batch = {}
        prot_masks_batch = {}
        
        for _, row in batch.iterrows():
            prot_id = row[prot_id_col]
            prot_seq = row[prot_seq_col]

        
            if not isinstance(prot_seq, str):
                prot_seq = str(prot_seq)

   
            prot_seq = prot_seq.strip()


            if not prot_seq:
                continue


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
            
            # Save attention mask
            prot_masks_batch[prot_id] = prot_attention_mask.cpu().numpy()
            

            dummy_drug_input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)
            dummy_drug_attention_mask = torch.zeros((1, 1), dtype=torch.long).to(device)
            
            # Compute embeddings
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
    

    print("Processing mRNA sequences in batches...")
    drug_batches = [unique_drugs.iloc[i:i+batch_size] for i in range(0, len(unique_drugs), batch_size)]
    for batch_idx, batch in enumerate(tqdm(drug_batches)):

        if not isinstance(batch, pd.DataFrame):
            batch = pd.DataFrame(batch)
        drug_embeddings_batch = {}
        drug_masks_batch = {}
        
        for _, row in batch.iterrows():
            drug_id = row[drug_id_col]
            drug_seq = row[drug_seq_col]
            
 
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
                # Ensure data is on CPU
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                prot_embeddings.update(batch_data)
        elif batch_file.startswith("protein_masks_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
                # Ensure data is on CPU
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                prot_masks.update(batch_data)
    

    for batch_file in os.listdir(temp_dir):
        if batch_file.startswith("drug_embeddings_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
                # Ensure data is on CPU
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                drug_embeddings.update(batch_data)
        elif batch_file.startswith("drug_masks_batch_"):
            with open(os.path.join(temp_dir, batch_file), "rb") as f:
                batch_data = pickle.load(f)
                # Ensure data is on CPU
                for key, value in batch_data.items():
                    if isinstance(value, np.ndarray):
                        batch_data[key] = torch.from_numpy(value).cpu()
                drug_masks.update(batch_data)
    
    # Save final results
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
    
    print(f"Embedding and mask computation completed! Results saved to {output_dir}")
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
        args.batch_size
    )