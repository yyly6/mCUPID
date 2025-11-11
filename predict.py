"""
Predicting mRNA-molecule interactions using a pre-trained model
Input a test.csv file and precomputed embedding dictionaries, output prediction results
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                           precision_recall_curve, f1_score, precision_score, 
                           recall_score, matthews_corrcoef)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from mcupid_models import mCUPID


class DictToObj:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

# Dataset class, same as in the training script
class EmbeddingDataset(Dataset):
    def __init__(self, df, prot_embeddings, drug_embeddings, prot_id_col, drug_id_col, label_col):
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
                raise KeyError(f"Protein ID {prot_id} does not exist in embedding dictionary")
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
            print(f"Protein ID: {prot_id_val}")
            print(f"Drug ID: {drug_id_val}")
            raise

# Batch processing function
def collate_fn(batch):
    prot_embeds, drug_embeds, prot_masks, drug_masks, labels = zip(*batch)
    
    # Stack embeddings into batches
    prot_embeds = torch.stack(prot_embeds)
    drug_embeds = torch.stack(drug_embeds)
    
    prot_masks = torch.stack(prot_masks)
    drug_masks = torch.stack(drug_masks)
    
    labels = torch.tensor(labels, dtype=torch.float)
    
    return prot_embeds, drug_embeds, prot_masks, drug_masks, labels

# Load embedding data
def load_embeddings(prot_embed_path, prot_mask_path, drug_embed_path, drug_mask_path):
    print("\nLoading embedding data...")
    
    print("Loading protein embeddings...")
    with open(prot_embed_path, 'rb') as f:
        prot_embeddings = pickle.load(f)
    
    print("Loading protein masks...")
    with open(prot_mask_path, 'rb') as f:
        prot_masks = pickle.load(f)
    
    print("Loading drug embeddings...")
    with open(drug_embed_path, 'rb') as f:
        drug_embeddings = pickle.load(f)
    
    print("Loading drug masks...")
    with open(drug_mask_path, 'rb') as f:
        drug_masks = pickle.load(f)
    
    # Merge embeddings and masks
    print("Merging embeddings and masks...")
    
    for prot_id in prot_embeddings.keys():
        if prot_id in prot_masks:
            prot_embeddings[prot_id] = {
                'embedding': prot_embeddings[prot_id],
                'mask': prot_masks[prot_id]
            }
    
    for drug_id in drug_embeddings.keys():
        if drug_id in drug_masks:
            drug_embeddings[drug_id] = {
                'embedding': drug_embeddings[drug_id],
                'mask': drug_masks[drug_id]
            }
    
    return prot_embeddings, drug_embeddings

# Model evaluation
def evaluate_model(predictions, actuals):
    predictions_binary = np.array(predictions) > 0.5
    actuals = np.array(actuals)
    
    # Check unique labels
    unique_labels = np.unique(actuals)
    n_unique = len(unique_labels)
    
    print(f"\nDataset label analysis:")
    print(f"Unique labels: {unique_labels}")
    print(f"Number of labels: {n_unique}")
    print(f"Positive samples: {np.sum(actuals == 1)}")
    print(f"Negative samples: {np.sum(actuals == 0)}")
    
 
    accuracy = accuracy_score(actuals, predictions_binary)
    
    # Initialize all metrics
    auc = float('nan')
    aupr = float('nan')
    precision = float('nan')
    recall = float('nan')
    f1 = float('nan')
    mcc = float('nan')
    
    # Only calculate ROC-AUC when there are two classes
    if n_unique == 2:
        try:
            auc = roc_auc_score(actuals, predictions)
        except ValueError as e:
            print(f"Error calculating AUC: {e}")
            auc = float('nan')
    else:
        print("⚠️ Warning: Only one class, cannot calculate AUC")
    
    try:
        if n_unique == 2:
            aupr = average_precision_score(actuals, predictions)
        else:
            # Baseline AUPR for single-class case
            baseline_aupr = np.mean(actuals)  # Proportion of positive samples
            print(f"⚠️ Single-class case, baseline AUPR: {baseline_aupr:.4f}")
            aupr = baseline_aupr
    except ValueError as e:
        print(f"Error calculating AUPR: {e}")
        aupr = float('nan')
    

    try:
        if n_unique == 2:
            precision = precision_score(actuals, predictions_binary, zero_division='warn')
            recall = recall_score(actuals, predictions_binary, zero_division='warn')
            f1 = f1_score(actuals, predictions_binary, zero_division='warn')
        else:
            # Handling for single-class cases
            if np.all(actuals == 1):  # All positive samples
                precision = np.mean(predictions_binary)  # Proportion predicted as positive
                recall = 1.0 if np.any(predictions_binary) else 0.0
            else:  # All negative samples
                precision = 0.0 if np.any(predictions_binary) else 1.0
                recall = np.mean(1 - predictions_binary)  # Proportion predicted as negative
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            print(f"⚠️ Single-class case - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
    except Exception as e:
        print(f"Error calculating classification metrics: {e}")
        precision, recall, f1 = 0.0, 0.0, 0.0
    
 
    try:
        if n_unique == 2:
            mcc = matthews_corrcoef(actuals, predictions_binary)
        else:
            print("⚠️ Single-class case, MCC set to 0")
            mcc = 0.0
    except Exception as e:
        print(f"Error calculating MCC: {e}")
        mcc = 0.0
    
    # Generate result dictionary
    metrics = {
        'AUC (auROC)': auc,
        'AUPR': aupr,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc,
        'Unique_Labels': n_unique,
        'Positive_Samples': int(np.sum(actuals == 1)),
        'Negative_Samples': int(np.sum(actuals == 0))
    }
    
    return metrics

# Single model prediction
def predict_with_model(model_path, test_loader, device, show_progress=True):
    print(f"\nLoading model: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['args']
    
    # Create model and load weights
    model = mCUPID(
        prot_out_dim=model_args['prot_dim'], 
        disease_out_dim=model_args['drug_dim'], 
        args=DictToObj(model_args)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prediction
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            prot_embed, drug_embed, prot_mask, drug_mask, label = batch
            prot_embed = prot_embed.to(device)
            drug_embed = drug_embed.to(device)
            prot_mask = prot_mask.to(device)
            drug_mask = drug_mask.to(device)
            
            output = model(prot_embed, drug_embed, prot_mask, drug_mask)
            
            # Process model output dimensions to ensure correct addition to list
            output_values = output.squeeze().cpu().numpy()
            if output_values.ndim == 0:  # If it's a 0-dimensional array (scalar)
                output_values = [output_values.item()]
            elif output_values.ndim == 1:  # If it's a 1-dimensional array
                output_values = output_values.tolist()
            else:  # If it's a multi-dimensional array, flatten
                output_values = output_values.flatten().tolist()
            
            predictions.extend(output_values)
            actuals.extend(label.cpu().numpy())
    
    # Calculate evaluation metrics
    metrics = evaluate_model(predictions, actuals)
    
    # Print evaluation results
    if show_progress:
        print(f"\nModel evaluation results:")
        for metric, value in metrics.items():
            if metric in ['Unique_Labels', 'Positive_Samples', 'Negative_Samples']:
                print(f"{metric}: {value}")
            elif isinstance(value, float) and not np.isnan(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A (cannot calculate)")
    
    return predictions, actuals, metrics

# Ensemble prediction from multiple models
def ensemble_prediction(model_dir, test_loader, device):
    import glob
    
    # Get all model files
    model_files = glob.glob(os.path.join(model_dir, "model_fold_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    print(f"Found {len(model_files)} model files")
    
    all_predictions = []
    actuals = None
    fold_metrics = []
    
    # Predict with each model
    for i, model_file in enumerate(model_files):
        print(f"\nUsing model {i+1}/{len(model_files)}: {os.path.basename(model_file)}")
        fold_predictions, fold_actuals, fold_metric = predict_with_model(
            model_file, test_loader, device, show_progress=False
        )
        
        all_predictions.append(fold_predictions)
        if actuals is None:
            actuals = fold_actuals
        
        fold_metrics.append({
            'fold': i + 1,
            **fold_metric
        })
        
        # Print current fold results
        print(f"Fold {i+1} evaluation results:")
        for metric, value in fold_metric.items():
            if metric in ['Unique_Labels', 'Positive_Samples', 'Negative_Samples']:
                print(f"{metric}: {value}")
            elif isinstance(value, float) and not np.isnan(value):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A")
    
    # Calculate average predictions
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Calculate ensemble model evaluation metrics
    ensemble_metrics = evaluate_model(ensemble_predictions, actuals)
    
    # Print fold results
    print("\n===== Results for all folds =====")
    for result in fold_metrics:
        fold_str = f"Fold {result['fold']}"
        metrics_strs = []
        for k, v in result.items():
            if k != 'fold':
                if k in ['Unique_Labels', 'Positive_Samples', 'Negative_Samples']:
                    metrics_strs.append(f"{k}: {v}")
                elif isinstance(v, float) and not np.isnan(v):
                    metrics_strs.append(f"{k}: {v:.4f}")
                else:
                    metrics_strs.append(f"{k}: N/A")
        metrics_str = ", ".join(metrics_strs)
        print(f"{fold_str}: {metrics_str}")
    
    # Print ensemble results
    print("\n===== Ensemble model results =====")
    for metric, value in ensemble_metrics.items():
        if metric in ['Unique_Labels', 'Positive_Samples', 'Negative_Samples']:
            print(f"{metric}: {value}")
        elif isinstance(value, float) and not np.isnan(value):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: N/A (cannot calculate)")
    
    if 'AUC (auROC)' in ensemble_metrics and not np.isnan(ensemble_metrics['AUC (auROC)']):
        print(f"\nNote - Area under ROC curve (auROC): {ensemble_metrics['AUC (auROC)']:.4f}")
    else:
        print(f"\nNote: Due to label distribution issues, cannot calculate valid auROC")
    
    return ensemble_predictions, actuals, ensemble_metrics, fold_metrics

# Save prediction results
def save_predictions(df, predictions, output_file):

    result_df = df.copy()
    result_df['prediction_score'] = predictions
    result_df['prediction_label'] = (np.array(predictions) > 0.5).astype(int)
    
    # Save results
    result_df.to_csv(output_file, index=False)
    print(f"\nPrediction results saved to: {output_file}")
    
    return result_df

# Plot ROC curve
def plot_roc_curve(y_true, y_pred, output_file=None):
    """Plot ROC curve and calculate AUC value"""
    
    # Check label distribution
    unique_labels = np.unique(y_true)
    
    if len(unique_labels) < 2:
        print(f"⚠️ Warning: Only one label class {unique_labels}, cannot plot ROC curve")
        if output_file:
            # Create an information chart
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f'Cannot plot ROC curve\nOnly one label class: {unique_labels[0]}\nPositive samples: {np.sum(y_true == 1)}\nNegative samples: {np.sum(y_true == 0)}', 
                    horizontalalignment='center', verticalalignment='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Cannot plot (single-class data)')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Information chart saved to: {output_file}")
            plt.close()
        return float('nan')
    
    try:
        # Calculate ROC curve points
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save figure or display
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {output_file}")
        else:
            plt.show()
        
        plt.close()
        return roc_auc
        
    except Exception as e:
        print(f"Error plotting ROC curve: {str(e)}")
        return float('nan')


def main():
    parser = argparse.ArgumentParser(description="Predict RNA-RNA interactions using a pre-trained model")
    
    # Data-related parameters
    parser.add_argument("--test_csv", type=str, required=True, 
                      help="Test data CSV file path")
    parser.add_argument("--prot_embed_path", type=str, required=True,
                      help="Precomputed protein embedding pickle file path")
    parser.add_argument("--prot_mask_path", type=str, required=True,
                      help="Precomputed protein attention mask pickle file path")
    parser.add_argument("--drug_embed_path", type=str, required=True,
                      help="Precomputed drug embedding pickle file path")
    parser.add_argument("--drug_mask_path", type=str, required=True,
                      help="Precomputed drug attention mask pickle file path")
    parser.add_argument("--prot_id_col", type=str, default="miRNA_ID",
                      help="Protein ID column name in CSV")
    parser.add_argument("--drug_id_col", type=str, default="mRNA_ID",
                      help="Drug ID column name in CSV")
    parser.add_argument("--label_col", type=str, default="label",
                      help="Label column name in CSV")
    
    # Model-related parameters
    parser.add_argument("--model_path", type=str, default=None,
                      help="Single model file path")
    parser.add_argument("--model_dir", type=str, default=None,
                      help="Model directory (for ensemble prediction)")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size")
    
    # Output-related
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                      help="Prediction results output file path")
    parser.add_argument("--plot_roc", action="store_true",
                      help="Whether to plot ROC curve")
    parser.add_argument("--roc_file", type=str, default="roc_curve.png",
                      help="ROC curve save path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Computing device")
    
    args = parser.parse_args()
    
    # Check model parameters
    if args.model_path is None and args.model_dir is None:
        raise ValueError("Must provide either --model_path or --model_dir parameter")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test data: {len(test_df)} records")
    
    # Load embedding data
    prot_embeddings, drug_embeddings = load_embeddings(
        args.prot_embed_path, args.prot_mask_path, args.drug_embed_path, args.drug_mask_path
    )
    
    # Create test dataset and data loader
    test_dataset = EmbeddingDataset(
        test_df, prot_embeddings, drug_embeddings, 
        args.prot_id_col, args.drug_id_col, args.label_col
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    # Make predictions
    if args.model_path is not None:
        # Use single model
        predictions, actuals, metrics = predict_with_model(
            args.model_path, test_loader, device
        )
    else:
        # Use ensemble model
        predictions, actuals, metrics, _ = ensemble_prediction(
            args.model_dir, test_loader, device
        )
    
    # Save prediction results
    result_df = save_predictions(test_df, predictions, args.output_file)
    
    # Print classification statistics
    if actuals is not None:
        pred_labels = (np.array(predictions) > 0.5).astype(int)
        actual_labels = np.array(actuals).astype(int)
        
        # Calculate confusion matrix
        TP = np.sum((pred_labels == 1) & (actual_labels == 1))
        TN = np.sum((pred_labels == 0) & (actual_labels == 0))
        FP = np.sum((pred_labels == 1) & (actual_labels == 0))
        FN = np.sum((pred_labels == 0) & (actual_labels == 1))
        
        print("\n===== Classification statistics =====")
        print(f"Total test samples: {len(actuals)}")
        print(f"True Positive (TP): {TP}")
        print(f"True Negative (TN): {TN}")
        print(f"False Positive (FP): {FP}")
        print(f"False Negative (FN): {FN}")
        print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.4f}")
        
        # Plot ROC curve
        if args.plot_roc:
            try:
                roc_auc = plot_roc_curve(actual_labels, predictions, args.roc_file)
                if not np.isnan(roc_auc):
                    print(f"Area under ROC curve (auROC): {roc_auc:.4f}")
                else:
                    print(f"Cannot calculate valid auROC due to data label distribution issues")
            except Exception as e:
                print(f"Error plotting ROC curve: {str(e)}")
    else:
        print("\n===== Classification statistics =====")
        print("Cannot calculate classification statistics because actuals is None")
        
        # Plot ROC curve
        if args.plot_roc:
            print("Cannot plot ROC curve because actuals is None")
    

    return result_df, predictions, metrics

if __name__ == "__main__":
    main()