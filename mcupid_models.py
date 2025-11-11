import logging
import os
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast_mode
from torch.nn import Module
from tqdm import tqdm
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

class mCUPID(nn.Module):
    def __init__(self, prot_out_dim, disease_out_dim, args):
        super(mCUPID, self).__init__()
        self.fusion = args.fusion
        self.drug_reg = nn.Linear(disease_out_dim, 768)
        self.prot_reg = nn.Linear(prot_out_dim, 768)

        if self.fusion == "CAN":
            self.can_layer = CAN_Layer(hidden_dim=768, num_heads=8, args=args)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=1536)
        elif self.fusion == "None":
            
            self.prot_reduce = nn.Linear(prot_out_dim, args.reduction_dim)
            self.drug_reduce = nn.Linear(disease_out_dim, args.reduction_dim)
            self.mlp_classifier = MlPdecoder_CAN(input_dim=args.reduction_dim * 2)

    def forward(self, prot_embed, drug_embed, prot_mask, drug_mask):
        if self.fusion == "None":
           
            prot_embed = self.prot_reduce(prot_embed.mean(1).mean(1))  # [batch_size, reduction_dim]
            drug_embed = self.drug_reduce(drug_embed.mean(1).mean(1))  # [batch_size, reduction_dim]
            joint_embed = torch.cat([prot_embed, drug_embed], dim=1)  # [batch_size, 2*reduction_dim]
        else:
            prot_embed = self.prot_reg(prot_embed)
            drug_embed = self.drug_reg(drug_embed)

            if self.fusion == "CAN":
                joint_embed = self.can_layer(prot_embed, drug_embed, prot_mask, drug_mask)
                # Save attention maps for visualization analysis
                # Automatically select the correct attention direction based on task type
                # Default to alpha_pd (miRNA-mRNA analysis)
                self.attention_map = getattr(self.can_layer, 'alpha_pd', None)
          
            else:
                # If not a supported fusion type, simply concatenate embeddings
                joint_embed = torch.cat([prot_embed, drug_embed], dim=1)

        score = self.mlp_classifier(joint_embed)
        return score

class CAN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Layer, self).__init__()
        self.agg_mode = args.agg_mode
        self.group_size = args.group_size  
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        # Check input dimensions
        if len(x.shape) == 4:  # [batch_size, batch, seq_len, hidden_dim]
            N, B, L, D = x.shape
            x = x.reshape(N * B, L, D)  
        else:
            N, L, D = x.shape
            
        groups = L // group_size
        
        # Process embeddings
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)  # Add a dimension (N, L) -> (N, L, 1)
        elif len(mask.shape) == 3:  # [batch_size, batch, seq_len]
            N, B, L = mask.shape
            mask = mask.reshape(N * B, L, 1)  
            
        mask_grouped = mask.view(N, groups, group_size, -1).any(dim=2)
        
        return x_grouped, mask_grouped

    def forward(self, protein, drug, mask_prot, mask_drug):
    
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        drug_grouped, mask_drug_grouped = self.group_embeddings(drug, mask_drug, self.group_size)

        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug_grouped), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug_grouped), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug_grouped), self.num_heads, self.head_size)


        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)

        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_drug_grouped)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug_grouped, mask_drug_grouped)

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2

    
        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            # Ensure mask_grouped has the correct dimensions
            if len(mask_prot_grouped.shape) == 3:
                mask_prot_grouped = mask_prot_grouped.squeeze(-1)  # [batch, groups, 1] -> [batch, groups]
            if len(mask_drug_grouped.shape) == 3:
                mask_drug_grouped = mask_drug_grouped.squeeze(-1)  # [batch, groups, 1] -> [batch, groups]
            
            prot_embed = (prot_embedding * mask_prot_grouped.unsqueeze(-1)).sum(1) / mask_prot_grouped.sum(-1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug_grouped.unsqueeze(-1)).sum(1) / mask_drug_grouped.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        query_embed = torch.cat([prot_embed, drug_embed], dim=1)
        
        # Save attention maps for analysis
        # For UTR-Protein analysis: Save drug(UTR) attention to protein
        # For miRNA-mRNA analysis: Save protein(miRNA) attention to drug(mRNA)  
        self.alpha_pd = alpha_pd  # Protein attention to drug (for miRNA analysis)
        self.alpha_dp = alpha_dp  # Drug attention to protein (for UTR analysis)
        
        return query_embed

class MlPdecoder_CAN(nn.Module):
    def __init__(self, input_dim):
        super(MlPdecoder_CAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.LayerNorm(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn3 = nn.LayerNorm(input_dim // 4)
        self.output = nn.Linear(input_dim // 4, 1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.bn3(torch.relu(self.fc3(x)))
        x = torch.sigmoid(self.output(x))
        return x



class BatchFileDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        batch_file = self.file_list[idx]
        data = torch.load(batch_file)
        return data['prot'], data['drug'], data['prot_mask'], data['drug_mask'], data['y']