"""
RNA-Drug Target Interaction Prediction Based on Precomputed Embeddings
Training and Evaluation Using K-Fold Cross Validation
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import numpy as np
import random  
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


def set_random_seed(seed):
    """Set random seed to ensure reproducible results"""
    torch.manual_seed(seed)
    np.random.seed(seed)
  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                            precision_recall_curve, f1_score, precision_score, 
                            recall_score, matthews_corrcoef)

from mcupid_models import mCUPID, CAN_Layer, MlPdecoder_CAN


def parse_config():
    parser = argparse.ArgumentParser(description="RNA-Drug Target Interaction Prediction Based on Precomputed Embeddings")
    
    # Data related parameters
    parser.add_argument("--data_dir", type=str, required=True, 
                      help="Data directory, should contain files from train_fold_0.csv to train_fold_4.csv")
    parser.add_argument("--prot_embed_path", type=str, required=True,
                      help="Precomputed RNA embedding pickle file path")
    parser.add_argument("--prot_mask_path", type=str, required=True,
                      help="Precomputed RNA attention mask pickle file path")
    parser.add_argument("--drug_embed_path", type=str, required=True,
                      help="Precomputed drug embedding pickle file path")
    parser.add_argument("--drug_mask_path", type=str, required=True,
                      help="Precomputed drug attention mask pickle file path")
    parser.add_argument("--prot_id_col", type=str, default="RNA_ID",
                      help="RNA ID column name in CSV")
    parser.add_argument("--drug_id_col", type=str, default="Drug_ID",
                      help="Drug ID column name in CSV")
    parser.add_argument("--label_col", type=str, default="label",
                      help="Label column name in CSV")
    
    # Model related parameters
    parser.add_argument("--prot_dim", type=int, default=768,
                      help="RNA embedding dimension")
    parser.add_argument("--drug_dim", type=int, default=768,
                      help="Drug embedding dimension")
    parser.add_argument("--fusion", type=str, default="CAN", choices=["CAN", "None"],
                      help="Fusion method: CAN or None (no fusion, direct concatenation)")
    parser.add_argument("--agg_mode", type=str, default="mean_all_tok", 
                      help="Aggregation mode")
    parser.add_argument("--group_size", type=int, default=1,
                      help="Group size in CAN fusion")
    
    # Training related parameters
    parser.add_argument("--batch_size", type=int, default=128,
                      help="Training batch size")
    parser.add_argument("--n_folds", type=int, default=5,
                      help="Number of cross-validation folds")
    parser.add_argument("--num_epochs", type=int, default=200,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                      help="Weight decay")
    parser.add_argument("--patience", type=int, default=10,
                      help="Patience value for early stopping")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate")
    
    # Other parameters
    parser.add_argument("--save_dir", type=str, default="./saved_models",
                      help="Model save directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Training device")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--ind_test_path", type=str, default=None,
                     help="Independent test set CSV file path (optional)")

    parser.add_argument("--reduction_dim", type=int, default=128,
                      help="Dimension reduction target dimension for RNA and drug embeddings when fusion is None")
    
    return parser.parse_args()


class EmbeddingDataset(Dataset):
    """Dataset class for loading precomputed embeddings"""
    
    def __init__(self, df, prot_embeddings, drug_embeddings, prot_id_col, drug_id_col, label_col):
        """
        Args:
            df: DataFrame containing RNA_ID, Drug_ID, and labels
            prot_embeddings: Dictionary mapping RNA_ID to embeddings
            drug_embeddings: Dictionary mapping Drug_ID to embeddings
            prot_id_col: RNA_ID column name in DataFrame
            drug_id_col: Drug_ID column name in DataFrame
            label_col: Label column name in DataFrame
        """
        self.df = df
        self.prot_embeddings = prot_embeddings
        self.drug_embeddings = drug_embeddings
        self.prot_id_col = prot_id_col
        self.drug_id_col = drug_id_col
        self.label_col = label_col
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            prot_id = str(row[self.prot_id_col])
            drug_id = str(row[self.drug_id_col])
            label = float(row[self.label_col])
            
         
            prot_data = self.prot_embeddings.get(prot_id, None)
            drug_data = self.drug_embeddings.get(drug_id, None)
            
            if prot_data is None:
                raise KeyError(f"RNA ID {prot_id} does not exist in embedding dictionary")
            if drug_data is None:
                raise KeyError(f"Drug ID {drug_id} does not exist in embedding dictionary")
                
      
            prot_embed = prot_data['embedding']
            prot_mask = torch.tensor(prot_data['mask'], dtype=torch.bool)
            
            drug_embed = drug_data['embedding']
            drug_mask = torch.tensor(drug_data['mask'], dtype=torch.bool)
            
       
            if len(prot_embed.shape) == 2:  # [seq_len, hidden_dim]
                prot_embed = prot_embed.reshape(1, *prot_embed.shape)
            if len(drug_embed.shape) == 2:  # [seq_len, hidden_dim]
                drug_embed = drug_embed.reshape(1, *drug_embed.shape)
                
       
            if len(prot_mask.shape) == 1:  # [seq_len]
                prot_mask = prot_mask.reshape(1, -1)
            if len(drug_mask.shape) == 1:  # [seq_len]
                drug_mask = drug_mask.reshape(1, -1)
            
    
            prot_embed = torch.tensor(prot_embed, dtype=torch.float)
            drug_embed = torch.tensor(drug_embed, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.float)
            
            return prot_embed, drug_embed, prot_mask, drug_mask, label
            
        except Exception as e:
            print(f"Error processing sample {idx}:")
            print(f"Error message: {str(e)}")
            prot_id_val = locals().get('prot_id', 'N/A')
            drug_id_val = locals().get('drug_id', 'N/A')
            print(f"RNA ID: {prot_id_val}")
            print(f"Drug ID: {drug_id_val}")
            raise


def collate_fn(batch):
    """Custom collate_fn to handle embeddings of different lengths"""
    prot_embeds, drug_embeds, prot_masks, drug_masks, labels = zip(*batch)
    

    prot_embeds = torch.stack(prot_embeds)
    drug_embeds = torch.stack(drug_embeds)
    
 
    prot_masks = torch.stack(prot_masks)
    drug_masks = torch.stack(drug_masks)
    

    labels = torch.tensor(labels, dtype=torch.float)
    
    return prot_embeds, drug_embeds, prot_masks, drug_masks, labels


def load_data_and_embeddings(args, fold):
    """Load data and precomputed embeddings"""
    import time
    from tqdm import tqdm
    
    print(f"Loading data and precomputed embeddings for fold {fold}...")
    start_time = time.time()
    
    
    print("Loading CSV files...")
    csv_start_time = time.time()
    train_df = pd.read_csv(os.path.join(args.data_dir, f"train_fold_{fold}.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, f"test_fold_{fold}.csv"))
    csv_time = time.time() - csv_start_time
    print(f"CSV files loaded successfully, time taken: {csv_time:.2f} seconds")
    
    
    print("\nLoading RNA embeddings...")
    prot_embed_start_time = time.time()
    with open(args.prot_embed_path, 'rb') as f:
        prot_embeddings = pickle.load(f)
    prot_embed_time = time.time() - prot_embed_start_time
    print(f"RNA embeddings loaded successfully, time taken: {prot_embed_time:.2f} seconds")
    
    print("\nLoading RNA masks...")
    prot_mask_start_time = time.time()
    with open(args.prot_mask_path, 'rb') as f:
        prot_masks = pickle.load(f)
    prot_mask_time = time.time() - prot_mask_start_time
    print(f"RNA masks loaded successfully, time taken: {prot_mask_time:.2f} seconds")
    
    print("\nLoading drug embeddings...")
    drug_embed_start_time = time.time()
    with open(args.drug_embed_path, 'rb') as f:
        drug_embeddings = pickle.load(f)
    drug_embed_time = time.time() - drug_embed_start_time
    print(f"Drug embeddings loaded successfully, time taken: {drug_embed_time:.2f} seconds")
    
    print("\nLoading drug masks...")
    drug_mask_start_time = time.time()
    with open(args.drug_mask_path, 'rb') as f:
        drug_masks = pickle.load(f)
    drug_mask_time = time.time() - drug_mask_start_time
    print(f"Drug masks loaded successfully, time taken: {drug_mask_time:.2f} seconds")
    

    print("\nMerging embeddings and masks...")
    merge_start_time = time.time()
    
    print("Processing RNA data...")
    prot_ids = list(prot_embeddings.keys())
    for prot_id in tqdm(prot_ids, desc="Merging RNA embeddings and masks"):
        if prot_id in prot_masks:
            prot_embeddings[prot_id] = {
                'embedding': prot_embeddings[prot_id],
                'mask': prot_masks[prot_id]
            }
        else:
            print(f"Warning: RNA ID {prot_id} does not exist in masks")
    
    print("\nProcessing drug data...")
    drug_ids = list(drug_embeddings.keys())
    for drug_id in tqdm(drug_ids, desc="Merging drug embeddings and masks"):
        if drug_id in drug_masks:
            drug_embeddings[drug_id] = {
                'embedding': drug_embeddings[drug_id],
                'mask': drug_masks[drug_id]
            }
        else:
            print(f"Warning: Drug ID {drug_id} does not exist in masks")
    
    merge_time = time.time() - merge_start_time
    print(f"Merge completed, time taken: {merge_time:.2f} seconds")
    

    print("\nValidating data integrity...")
    missing_prots = []
    missing_drugs = []
    
    for _, row in tqdm(train_df.iterrows(), desc="Validating training set", total=len(train_df)):
        prot_id = str(row[args.prot_id_col])
        drug_id = str(row[args.drug_id_col])
        
        if prot_id not in prot_embeddings:
            missing_prots.append(prot_id)
        if drug_id not in drug_embeddings:
            missing_drugs.append(drug_id)
    
    for _, row in tqdm(test_df.iterrows(), desc="Validating test set", total=len(test_df)):
        prot_id = str(row[args.prot_id_col])
        drug_id = str(row[args.drug_id_col])
        
        if prot_id not in prot_embeddings:
            missing_prots.append(prot_id)
        if drug_id not in drug_embeddings:
            missing_drugs.append(drug_id)
    
    missing_prots = list(set(missing_prots))
    missing_drugs = list(set(missing_drugs))
    
    if missing_prots:
        print(f"Warning: {len(missing_prots)} RNA IDs do not exist in embeddings")
        if len(missing_prots) < 10:
            print(f"Missing RNA IDs: {missing_prots}")
        else:
            print(f"Some missing RNA IDs: {missing_prots[:10]}...")
    
    if missing_drugs:
        print(f"Warning: {len(missing_drugs)} drug IDs do not exist in embeddings")
        if len(missing_drugs) < 10:
            print(f"Missing drug IDs: {missing_drugs}")
        else:
            print(f"Some missing drug IDs: {missing_drugs[:10]}...")
    
   
    total_time = time.time() - start_time
    print("\n===== Data Loading Summary =====")
    print(f"Total loading time: {total_time:.2f} seconds")
    print(f"CSV file loading time: {csv_time:.2f} seconds ({csv_time/total_time*100:.1f}%)")
    print(f"RNA embedding loading time: {prot_embed_time:.2f} seconds ({prot_embed_time/total_time*100:.1f}%)")
    print(f"RNA mask loading time: {prot_mask_time:.2f} seconds ({prot_mask_time/total_time*100:.1f}%)")
    print(f"Drug embedding loading time: {drug_embed_time:.2f} seconds ({drug_embed_time/total_time*100:.1f}%)")
    print(f"Drug mask loading time: {drug_mask_time:.2f} seconds ({drug_mask_time/total_time*100:.1f}%)")
    print(f"Data merging time: {merge_time:.2f} seconds ({merge_time/total_time*100:.1f}%)")
    print(f"Loaded {len(train_df)} training data and {len(test_df)} test data")
    print(f"RNA embeddings: {len(prot_embeddings)}, drug embeddings: {len(drug_embeddings)}")
    
    if missing_prots or missing_drugs:
        print("\nWarning: Missing IDs exist, please check data integrity!")
    
    return train_df, test_df, prot_embeddings, drug_embeddings


def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, args):
    """Train the model"""
    best_auc = 0.5  
    best_model_state = model.state_dict()  
    epochs_without_improvement = 0
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            prot_embed, drug_embed, prot_mask, drug_mask, label = batch
            prot_embed = prot_embed.to(device)
            drug_embed = drug_embed.to(device)
            prot_mask = prot_mask.to(device)
            drug_mask = drug_mask.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(prot_embed, drug_embed, prot_mask, drug_mask)
            loss = criterion(output, label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        predictions, actuals = [], []
        
        with torch.no_grad():
            for batch in valid_loader:
                prot_embed, drug_embed, prot_mask, drug_mask, label = batch
                prot_embed = prot_embed.to(device)
                drug_embed = drug_embed.to(device)
                prot_mask = prot_mask.to(device)
                drug_mask = drug_mask.to(device)
                
                output = model(prot_embed, drug_embed, prot_mask, drug_mask)
                predictions.extend(np.atleast_1d(output.squeeze().cpu().numpy()))
                actuals.extend(label.cpu().numpy())
            
            # Calculate AUC on validation set
            try:
                auc = roc_auc_score(actuals, predictions)
                print(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Validation AUC: {auc:.4f}')
                
                # Save best model
                if auc > best_auc:
                    best_auc = auc
                    best_model_state = model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= args.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            except ValueError:
                print(f'Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Validation AUC: N/A (all predictions in one class)')
    
    return best_model_state, best_auc


def evaluate(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            prot_embed, drug_embed, prot_mask, drug_mask, label = batch
            prot_embed = prot_embed.to(device)
            drug_embed = drug_embed.to(device)
            prot_mask = prot_mask.to(device)
            drug_mask = drug_mask.to(device)
            
            output = model(prot_embed, drug_embed, prot_mask, drug_mask)
            predictions.extend(np.atleast_1d(output.squeeze().cpu().numpy()))
            actuals.extend(label.cpu().numpy())
    
    predictions_binary = np.array(predictions) > 0.5
    actuals = np.array(actuals)
    

    try:
        auc = roc_auc_score(actuals, predictions)
    except ValueError:
        auc = float('nan')
    
    try:
        aupr = average_precision_score(actuals, predictions)
    except ValueError:
        aupr = float('nan')
    
    accuracy = accuracy_score(actuals, predictions_binary)
    
    try:
        precision = precision_score(actuals, predictions_binary)
        recall = recall_score(actuals, predictions_binary)
        f1 = f1_score(actuals, predictions_binary)
        mcc = matthews_corrcoef(actuals, predictions_binary)
    except:
        precision, recall, f1, mcc = 0, 0, 0, 0
    
    return {
        'AUC': auc,
        'AUPR': aupr,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    }


def select_best_model(metrics_history):

    combined_scores = []
    for epoch_metrics in metrics_history:
        combined_score = (
            0.4 * epoch_metrics['AUC'] + 
            0.3 * epoch_metrics['AUPR'] + 
            0.2 * epoch_metrics['F1'] + 
            0.1 * epoch_metrics['MCC']
        )
        combined_scores.append(combined_score)
    
    best_epoch = np.argmax(combined_scores)
    return best_epoch


def k_fold_cross_validation(args):
    """Perform K-fold cross validation"""
  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
   
    fold_results = []

    all_test_predictions = []
    test_ids = None  
    
    # If there is an independent test set, pre-load it
    ind_test_df = None
    if args.ind_test_path is not None and os.path.isfile(args.ind_test_path):
        print(f"Loading independent test set: {args.ind_test_path}")
        ind_test_df = pd.read_csv(args.ind_test_path)
    ind_test_metrics_all = []
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Perform K-fold cross validation
    for fold in range(args.n_folds):
        print(f"\n===== Fold {fold+1}/{args.n_folds} =====")
        
        try:
            # Load data for current fold
            train_df, test_df, prot_embeddings, drug_embeddings = load_data_and_embeddings(args, fold)

            # Select appropriate validation set splitting method based on startup type
            if args.setting == 'RNA_coldstart':
                # RNA cold start: Ensure RNA in validation set is not in training set
                # Extract RNA IDs from training set
                train_rna_ids = set(train_df[args.prot_id_col].unique())
                # Calculate required validation set data size (20% of training set data)
                val_data_size = int(len(train_df) * 0.2)
                
                # Group by RNA ID and calculate data size per group
                grouped = train_df.groupby(args.prot_id_col)
                id_data_counts = grouped.size()
                # Convert to series and sort
                id_data_counts = pd.Series(id_data_counts).sort_values(ascending=False, kind='quicksort')
                
                # Select validation set IDs until target data size is reached
                val_rna_ids = []
                selected_data_count = 0
                for rna_id, count in id_data_counts.items():
                    if selected_data_count >= val_data_size:
                        break
                    val_rna_ids.append(rna_id)
                    selected_data_count += count
                
                # Build validation set
                val_df = train_df[train_df[args.prot_id_col].isin(val_rna_ids)].copy()
                # Remove validation set samples from training set
                train_df = train_df[~train_df[args.prot_id_col].isin(val_rna_ids)].copy()
            elif args.setting == 'Drug_coldstart':
                # Drug cold start: Similar logic to RNA cold start
                train_drug_ids = set(train_df[args.drug_id_col].unique())
                val_data_size = int(len(train_df) * 0.2)
                
                grouped = train_df.groupby(args.drug_id_col)
                id_data_counts = grouped.size()
           
                id_data_counts = pd.Series(id_data_counts).sort_values(ascending=False, kind='quicksort')
                
                val_drug_ids = []
                selected_data_count = 0
                for drug_id, count in id_data_counts.items():
                    if selected_data_count >= val_data_size:
                        break
                    val_drug_ids.append(drug_id)
                    selected_data_count += count
                
                val_df = train_df[train_df[args.drug_id_col].isin(val_drug_ids)].copy()
                train_df = train_df[~train_df[args.drug_id_col].isin(val_drug_ids)].copy()
            else:
                # Warm start: Ensure IDs do not overlap and data is split 8:2
                if args.setting == 'warm_start_by_prot':
                    group_col = args.prot_id_col
                elif args.setting == 'warm_start_by_drug':
                    group_col = args.drug_id_col
                else:
                    group_col = args.prot_id_col
                
                # Group by ID and calculate data size per group
                grouped = train_df.groupby(group_col)
                id_data_counts = grouped.size()
                
                # Split IDs by data size proportion
                total_data = len(train_df)
                val_data_size = int(total_data * 0.2)  # 20% for validation
                
                # Randomly shuffle IDs
                all_ids = list(id_data_counts.index)
                random.shuffle(all_ids)
                
                # Select validation set IDs until target data size is reached
                val_ids = []
                selected_data_count = 0
                for id in all_ids:
                    if selected_data_count >= val_data_size:
                        break
                    val_ids.append(id)
                    selected_data_count += id_data_counts[id]
                
                # Build validation set and training set
                val_df = train_df[train_df[group_col].isin(val_ids)].copy()
                # Remove validation set samples from training set
                train_df = train_df[~train_df[group_col].isin(val_ids)].copy()
            print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}, Test set size: {len(test_df)}")

         
            train_dataset = EmbeddingDataset(
                train_df, prot_embeddings, drug_embeddings, 
                args.prot_id_col, args.drug_id_col, args.label_col
            )
            val_dataset = EmbeddingDataset(
                val_df, prot_embeddings, drug_embeddings, 
                args.prot_id_col, args.drug_id_col, args.label_col
            )
            test_dataset = EmbeddingDataset(
                test_df, prot_embeddings, drug_embeddings, 
                args.prot_id_col, args.drug_id_col, args.label_col
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                collate_fn=collate_fn, 
                num_workers=0, 
                pin_memory=True,
                drop_last=False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                collate_fn=collate_fn, 
                num_workers=0, 
                pin_memory=True,
                drop_last=False
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                collate_fn=collate_fn, 
                num_workers=0, 
                pin_memory=True,
                drop_last=False
            )

            # Initialize model
            model = mCUPID(
                prot_out_dim=args.prot_dim,  
                disease_out_dim=args.drug_dim,  
                args=args
            ).to(device)

            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-8)

            # Train model (use validation set for early stopping)
            best_model_state, best_val_auc = train(
                model, train_loader, val_loader, criterion, optimizer, scheduler, device, args
            )

            # Load best model parameters
            model.load_state_dict(best_model_state)

            # Evaluate on test set
            test_metrics = evaluate(model, test_loader, device)
            
            # Save model
            model_save_path = os.path.join(args.save_dir, f"model_fold_{fold+1}.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'val_auc': best_val_auc,
                'test_metrics': test_metrics,
                'args': vars(args)
            }, model_save_path)
            
            # Save test set predictions
            model.eval()
            test_predictions = []
            current_test_ids = []
            
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    prot_embed, drug_embed, prot_mask, drug_mask, _ = batch
                    prot_embed = prot_embed.to(device)
                    drug_embed = drug_embed.to(device)
                    prot_mask = prot_mask.to(device)
                    drug_mask = drug_mask.to(device)
                    
                    output = model(prot_embed, drug_embed, prot_mask, drug_mask)
                    test_predictions.extend(np.atleast_1d(output.squeeze().cpu().numpy()))
                    
                    start_idx = i * args.batch_size
                    end_idx = min(start_idx + args.batch_size, len(test_df))
                  
                    batch_ids = test_df.iloc[start_idx:end_idx][[args.prot_id_col, args.drug_id_col]].values.tolist()
                    current_test_ids.extend(batch_ids)
            
           
            fold_predictions = pd.DataFrame({
                args.prot_id_col: [x[0] for x in current_test_ids],
                args.drug_id_col: [x[1] for x in current_test_ids],
                'prediction': test_predictions
            })
            fold_predictions.to_csv(os.path.join(args.save_dir, f'test_predictions_fold_{fold+1}.csv'), index=False)
            print(f"Fold {fold+1} test predictions saved to {os.path.join(args.save_dir, f'test_predictions_fold_{fold+1}.csv')}")
            
           
            all_test_predictions.append(test_predictions)
            
        
            if test_ids is None:
                test_ids = current_test_ids
            
        
            fold_metric_result = {
                'fold': fold + 1,
                'val_auc': float(best_val_auc)
            }
            
            for metric, value in test_metrics.items():
          
                try:
                    if isinstance(value, complex):
                        converted_value = float(value.real)   
                    elif isinstance(value, (int, float, np.number)):
                        converted_value = float(value)
                    else:
                         
                        try:
                            converted_value = float(str(value))
                        except (ValueError, TypeError):
                            converted_value = float('nan')
                    fold_metric_result[metric] = converted_value
                except (TypeError, ValueError):
                    print(f"Warning: Unable to convert metric '{metric}' value to float, using NaN instead")
                    fold_metric_result[metric] = float('nan')
            
          
            fold_results.append(fold_metric_result)
            
            
            print(f"\nFold {fold+1} Test Results:")
            for metric, value in test_metrics.items():
            
                try:
                    if isinstance(value, complex):
                        converted_value = float(value.real)   
                    elif isinstance(value, (int, float, np.number)):
                        converted_value = float(value)
                    else:
                         
                        try:
                            converted_value = float(str(value))
                        except (ValueError, TypeError):
                            converted_value = float('nan')
                    print(f"{metric}: {converted_value:.4f}")
                except (TypeError, ValueError):
                    print(f"{metric}: NaN")
                
            # If there is an independent test set, evaluate and save metrics
            if ind_test_df is not None:
                print("Evaluating on independent test set...")
                ind_test_dataset = EmbeddingDataset(
                    ind_test_df, prot_embeddings, drug_embeddings,
                    args.prot_id_col, args.drug_id_col, args.label_col
                )
                ind_test_loader = DataLoader(
                    ind_test_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False
                )
                ind_metrics = evaluate(model, ind_test_loader, device)
                ind_test_metrics_all.append(ind_metrics)
                print(f"Fold {fold+1} Independent Test Set Results:")
                for metric, value in ind_metrics.items():
                   
                    try:
                        if isinstance(value, complex):
                            converted_value = float(value.real)   
                        elif isinstance(value, (int, float, np.number)):
                            converted_value = float(value)
                        else:
                             
                            try:
                                converted_value = float(str(value))
                            except (ValueError, TypeError):
                                converted_value = float('nan')
                        print(f"{metric}: {converted_value:.4f}")
                    except (TypeError, ValueError):
                        print(f"{metric}: NaN")
        except Exception as e:
            print(f"Error processing fold {fold+1}:")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()  
            print("Skipping current fold, continuing to next fold...")
            continue
    
  
    avg_results = {}
    if len(fold_results) > 0:
     
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())
        all_metrics.discard('fold')  
       
        for metric in all_metrics:
            try:
                
                valid_values = []
                for result in fold_results:
                    if metric in result and isinstance(result[metric], (int, float, np.number)) and not np.isnan(result[metric]):
                        valid_values.append(float(result[metric]))
                
                if valid_values:
                    avg_results[metric] = np.mean(valid_values)
                else:
                    avg_results[metric] = float('nan')
            except Exception as e:
                print(f"Error calculating average for metric '{metric}': {str(e)}")
                avg_results[metric] = float('nan')
    else:
        print("Warning: No valid fold results, unable to calculate average metrics")
    

    final_predictions = pd.DataFrame()
    if all_test_predictions and test_ids is not None:
        try:
            # Ensure all prediction arrays have the same length
            min_length = min(len(preds) for preds in all_test_predictions)
            
            # Truncate all prediction arrays to minimum length
            truncated_predictions = [preds[:min_length] for preds in all_test_predictions]
            
            # Calculate average
            avg_test_predictions = np.mean(truncated_predictions, axis=0)
            
            # Ensure test_ids length does not exceed average prediction length
            if len(test_ids) > len(avg_test_predictions):
                test_ids = test_ids[:len(avg_test_predictions)]
            
            # Create final prediction results DataFrame
            final_predictions = pd.DataFrame({
                args.prot_id_col: [x[0] for x in test_ids],
                args.drug_id_col: [x[1] for x in test_ids],
                'prediction': avg_test_predictions
            })
            
      
            final_predictions.to_csv(os.path.join(args.save_dir, 'test_predictions.csv'), index=False)
        except Exception as e:
            print(f"Error calculating or saving average predictions: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Warning: No test predictions or test IDs available")
    
    
    print("\n===== Results for All Folds =====")
    for result in fold_results:
        fold_str = f"Fold {result['fold']}"
        metrics_str = ", ".join([
            f"{k}: {v:.4f}" if isinstance(v, (int, float, np.number)) and not np.isnan(v) else f"{k}: NaN" 
            for k, v in result.items() if k != 'fold'
        ])
        print(f"{fold_str}: {metrics_str}")
    

    print("\n===== Average Results =====")
    for metric, value in avg_results.items():
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: NaN")
    

    if ind_test_metrics_all:
        ind_test_metrics_df = pd.DataFrame(ind_test_metrics_all)
        ind_test_metrics_df.to_csv(os.path.join(args.save_dir, 'ind_test_metrics.csv'), index=False)
        print(f"Independent test set metrics for all folds saved to {os.path.join(args.save_dir, 'ind_test_metrics.csv')}")
    
    return fold_results, avg_results, final_predictions


def main():
    args = parse_config()
    
    data_dir = args.data_dir.lower()
    if 'rna_coldstart' in data_dir or 'mrna_coldstart' in data_dir:
        args.setting = 'RNA_coldstart'
    elif 'drug_coldstart' in data_dir:
        args.setting = 'Drug_coldstart'
    else:
        args.setting = 'warm_start'
    
    print(f"Automatically detected setting type: {args.setting}")
    

    print("\n===== Training Configuration =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    

    set_random_seed(args.seed)

    fold_results, avg_results, final_predictions = k_fold_cross_validation(args)
    

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(args.save_dir, 'fold_results.csv'), index=False)
    
    avg_results_df = pd.DataFrame([avg_results])
    avg_results_df.to_csv(os.path.join(args.save_dir, 'avg_results.csv'), index=False)
    
    print(f"\nTraining completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
