import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import optuna
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

from sklearn.metrics import precision_recall_curve, roc_curve, auc

import time
import csv
import shap
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class SqueezeExcitation(nn.Module):
    def __init__(self, embed_dim, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction)
        self.fc2 = nn.Linear(embed_dim // reduction, embed_dim)

    def forward(self, x):
        z = x.mean(dim=1)
        z = torch.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))
        return x * z.unsqueeze(1)

class FTTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(FTTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.se = SqueezeExcitation(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.se(ffn_output)
        x = self.norm2(x + ffn_output)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FTTransformer(nn.Module):
    def __init__(self, num_features, num_classes, embed_dim, num_heads, ff_dim, num_layers, dropout_rate):
        super(FTTransformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.transformer_layers = nn.ModuleList([
            FTTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if len(x.shape) == 2:
            x = x.unsqueeze(1)  
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def calculate_global_network_attributes(ppi_df):
    import networkx as nx
    G = nx.Graph()
    for _, row in ppi_df.iterrows():
        G.add_edge(row['node1'], row['node2'], weight=row['combined_score'])

    degree_centrality = nx.degree_centrality(G)
    pagerank = nx.pagerank(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    return degree_centrality, pagerank, betweenness_centrality

def calculate_gene_impact_scores(ppi_df, increment=0.1):
    all_genes = set(ppi_df['node1']).union(set(ppi_df['node2']))
    impact_scores = {gene: 0 for gene in all_genes}
    for _, row in ppi_df.iterrows():
        gene1, gene2 = row['node1'], row['node2']
        impact_scores[gene1] += increment
        impact_scores[gene2] += increment
    return impact_scores

def calculate_patient_ppi_features(patient_mutations, impact_scores, global_attributes, genes):
    degree_centrality, pagerank, betweenness_centrality = global_attributes
    patient_ppi_features = np.array([
        (impact_scores.get(gene, 0) if patient_mutations.get(gene, 0) == 1 else 0) +
        degree_centrality.get(gene, 0) +
        pagerank.get(gene, 0) +
        betweenness_centrality.get(gene, 0)
        for gene in genes
    ])
    return patient_ppi_features

class GatedFeatureSelector(nn.Module):
    def __init__(self, input_dim, reduction=16, stages=3):
        super(GatedFeatureSelector, self).__init__()
        self.stages = stages
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // reduction),
                nn.ReLU(),
                nn.Linear(input_dim // reduction, input_dim),
                nn.Sigmoid()
            ) for _ in range(stages)
        ])

    def forward(self, x):
        for layer in self.layers:
            weights = layer(x.mean(dim=0))
            x = x * weights.unsqueeze(0)
        return x

def process_data_with_global_attributes(data_train, data_test, ppi_df, genes, device):
    impact_scores = calculate_gene_impact_scores(ppi_df)
    global_attributes = calculate_global_network_attributes(ppi_df)

    train_ppi_features = np.array([
        calculate_patient_ppi_features(patient[genes].to_dict(), impact_scores, global_attributes, genes)
        for _, patient in data_train.iterrows()
    ])
    test_ppi_features = np.array([
        calculate_patient_ppi_features(patient[genes].to_dict(), impact_scores, global_attributes, genes)
        for _, patient in data_test.iterrows()
    ])

    train_features = np.hstack([data_train.values, train_ppi_features])
    test_features = np.hstack([data_test.values, test_ppi_features])

    gated_selector = GatedFeatureSelector(train_features.shape[1]).to(device)
    gated_selector.eval()

    train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
    with torch.no_grad():
        train_features = gated_selector(train_features_tensor).cpu().numpy()
        test_features = gated_selector(test_features_tensor).cpu().numpy()

    return train_features, test_features

def train_classification_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train, total_train = 0, 0
        for x_batch, y_batch_class in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            x_batch, y_batch_class = x_batch.to(device), y_batch_class.to(device)
            optimizer.zero_grad()
            class_output = model(x_batch)
            loss = criterion(class_output, y_batch_class)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(class_output, dim=1)
            correct_train += (predicted == y_batch_class).sum().item()
            total_train += y_batch_class.size(0)

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        train_acc = correct_train / total_train if total_train > 0 else 0.0
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch_class in val_loader:
                x_batch, y_batch_class = x_batch.to(device), y_batch_class.to(device)
                class_output = model(x_batch)
                val_loss += criterion(class_output, y_batch_class).item()
                _, predicted = torch.max(class_output, dim=1)
                correct_val += (predicted == y_batch_class).sum().item()
                total_val += y_batch_class.size(0)

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0.0
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import shap


def extract_classification_features(model, test_loader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            class_output = model(x_batch)
            features.append(class_output.cpu().numpy())
    return np.vstack(features)
def feature_selection_on_both(train_features, test_features, data_train, threshold_variance=0.001, threshold_correlation=0.95):
   
    target_columns = ['OS_MONTHS', 'OS_STATUS']
    original_feature_names = data_train.columns.tolist()
  
    features_to_keep = [col for col in original_feature_names if col not in target_columns]
    
    train_features = train_features[:, [i for i, col in enumerate(original_feature_names) if col in features_to_keep]]
    test_features = test_features[:, [i for i, col in enumerate(original_feature_names) if col in features_to_keep]]

    
    feature_variances = np.var(train_features, axis=0)
    low_variance_indices = np.where(feature_variances < threshold_variance)[0]
    train_features = np.delete(train_features, low_variance_indices, axis=1)
    test_features = np.delete(test_features, low_variance_indices, axis=1)

    correlation_months = np.corrcoef(train_features.T, data_train['OS_MONTHS'])
    high_correlation_indices_months = np.where(np.abs(correlation_months[-1, :-1]) > threshold_correlation)[0]
    train_features = np.delete(train_features, high_correlation_indices_months, axis=1)
    test_features = np.delete(test_features, high_correlation_indices_months, axis=1)

    correlation_status = np.corrcoef(train_features.T, data_train['OS_STATUS'])
    high_correlation_indices_status = np.where(np.abs(correlation_status[-1, :-1]) > threshold_correlation)[0]
    train_features = np.delete(train_features, high_correlation_indices_status, axis=1)
    test_features = np.delete(test_features, high_correlation_indices_status, axis=1)

 
    remaining_feature_names = [features_to_keep[i] for i in range(len(features_to_keep)) if i not in low_variance_indices and i not in high_correlation_indices_months and i not in high_correlation_indices_status]
    
    
    feature_name_map = {remaining_feature_names[i]: i for i in range(len(remaining_feature_names))}
    print("Updated feature name map after feature selection:", feature_name_map)

   
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    return train_features, test_features, feature_name_map 

import shap
from lifelines.statistics import logrank_test
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from lifelines.plotting import add_at_risk_counts  


import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_km_curve(df, group_col, time_col, event_col, title, save_path):
   
    
    df_ = df.dropna(subset=[group_col, time_col, event_col])
    unique_groups = df_[group_col].unique()
    
    
    fig, ax = plt.subplots(figsize=(8, 6))  
    kmfs = []
    
    
    group_colors = {unique_groups[0]: 'skyblue', unique_groups[1]: 'orange'} if len(unique_groups) == 2 else {}
    
    group_labels = {unique_groups[0]: 'High Risk', unique_groups[1]: 'Low Risk'} if len(unique_groups) == 2 else {}

    for grp in unique_groups:
        sub = df_[df_[group_col] == grp]
        kmf = KaplanMeierFitter(label=group_labels.get(grp, str(grp)))  
        kmf.fit(durations=sub[time_col], event_observed=sub[event_col])
        
        
        color = group_colors.get(grp, 'gray')  
        kmf.plot(ci_show=False, ax=ax, color=color)
        kmfs.append(kmf)
    
    
    if len(unique_groups) == 2:
        grp1, grp2 = unique_groups[0], unique_groups[1]
        sub1 = df_[df_[group_col] == grp1]
        sub2 = df_[df_[group_col] == grp2]
        res = logrank_test(
            sub1[time_col], sub2[time_col],
            event_observed_A=sub1[event_col],
            event_observed_B=sub2[event_col]
        )
        p_val = res.p_value
        ax.text(
            0.65, 0.9,
            f"p = {p_val:.3e}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time (Months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.legend(title="Risk Group")  
    
   
    add_at_risk_counts(*kmfs, ax=ax)
    
    
    plt.tight_layout()
    
    
    plt.savefig(save_path)
    plt.show()

def remove_constant_features(df):
   
    numeric_df = df.select_dtypes(include=[np.number])

    constant_columns = [col for col in numeric_df.columns if numeric_df[col].std() == 0]
    if constant_columns:
        print(f"Removing constant features: {constant_columns}")
    
    df = df.drop(columns=constant_columns)
    return df


def perform_survival_analysis(classification_features,
                              os_months, os_status,
                              cancer_types,
                              model, test_loader, device,
                              cancer_name_map, feature_name_map, selected_feature_names):
    cox = CoxPHFitter(penalizer=0.1)
    
   
    df = pd.DataFrame(classification_features, columns=selected_feature_names)

   
    df["OS_MONTHS"] = os_months
    df["OS_STATUS"] = os_status
    df["Cancer_Type"] = cancer_types  # Cancer_Type used for later analysis

    df["Cancer_Type"] = df["Cancer_Type"].map(cancer_name_map)

    df.columns = selected_feature_names + ["OS_MONTHS", "OS_STATUS", "Cancer_Type"]
    
    df = remove_constant_features(df)

    print("DataFrame columns after mapping:", df.columns)
    df_features = df.drop(columns=["Cancer_Type"])  # Exclude the 'Cancer_Type' column
    cox.fit(df_features, duration_col="OS_MONTHS", event_col="OS_STATUS")
    
    mapped_params = {selected_feature_names[i]: coef
                     for i, coef in enumerate(cox.params_)}
    mapped_p_values = {selected_feature_names[i]: p_val
                       for i, p_val in enumerate(cox.summary['p'])}
    
    print(f"Mapped coefficients: {mapped_params}")
    print(f"Mapped p-values: {mapped_p_values}")

    print(f"Params: {mapped_params}")
    print(f"P-values: {mapped_p_values}")
    print(f"Features: {df_features.columns}")

    summary = cox.summary

    with open('cox_model_summary.csv', 'w') as f:
        summary.to_csv(f, index=True)


    print("Cox model summary saved as 'cox_model_summary.txt'.")
    print(f"Mapped params keys: {list(mapped_params.keys())}")
    print(f"Mapped p-values keys: {list(mapped_p_values.keys())}")
    
    # Check lengths
    print(f"Length of mapped_params: {len(mapped_params)}")
    print(f"Length of mapped_p_values: {len(mapped_p_values)}")

    coefficients = pd.DataFrame({
        'feature': list(mapped_params.keys()),  
        'coefficient': list(mapped_params.values()),
        'p_value': list(mapped_p_values.values())
    })
    coefficients.to_csv('cox_model_coefficients.csv', index=False)
    print("Cox model coefficients saved as 'cox_model_coefficients.csv'.")

    df["risk_score"] = cox.predict_partial_hazard(df_features)

    df["risk_group"] = "Low Risk"  # Default to low risk
    for cancer_type in df['Cancer_Type'].unique():
        cancer_data = df[df['Cancer_Type'] == cancer_type]
        
        # Calculate median risk score for the current cancer type
        median_risk_score = cancer_data['risk_score'].median()
        df.loc[df['Cancer_Type'] == cancer_type, 'risk_group'] = np.where(cancer_data['risk_score'] >= median_risk_score, 'High Risk', 'Low Risk')

    df[['Cancer_Type', 'OS_MONTHS', 'OS_STATUS', 'risk_score']].to_csv('risk_score_table.csv', index=False)
    print("Risk score table saved as 'risk_score_table.csv'.")

    for cancer_type in df['Cancer_Type'].unique():
        cancer_data = df[df['Cancer_Type'] == cancer_type]
        
        plt.figure(figsize=(8, 6))
        cancer_data['risk_score'].hist(bins=50, alpha=0.75)
        plt.title(f'Risk Score Distribution for {cancer_type}')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        plt.savefig(f'risk_score_distribution_{cancer_type}.png', dpi=300)
        plt.close()
        print(f"Risk score distribution for {cancer_type} saved as 'risk_score_distribution_{cancer_type}.png'.")

    kmf = KaplanMeierFitter()
    
    for cancer_type in df['Cancer_Type'].unique():
        cancer_data = df[df['Cancer_Type'] == cancer_type]
        
        plot_km_curve(cancer_data,
                      group_col='risk_group', 
                      time_col='OS_MONTHS', 
                      event_col='OS_STATUS', 
                      title=f'{cancer_type} Kaplan-Meier Curve', 
                      save_path=f'kaplan_meier_{cancer_type}.png')

    logrank_results = {}
    for cancer_type in df['Cancer_Type'].unique():
        cancer_data = df[df['Cancer_Type'] == cancer_type]
        
        results = logrank_test(
            cancer_data[cancer_data['risk_group'] == 'High Risk']['OS_MONTHS'],
            cancer_data[cancer_data['risk_group'] == 'Low Risk']['OS_MONTHS'],
            event_observed_A=cancer_data[cancer_data['risk_group'] == 'High Risk']['OS_STATUS'],
            event_observed_B=cancer_data[cancer_data['risk_group'] == 'Low Risk']['OS_STATUS']
        )
        p_value = results.p_value
        logrank_results[cancer_type] = p_value
        
        with open(f'logrank_test_results_{cancer_type}.txt', 'w') as f:
            f.write(f"Log-rank test p-value for {cancer_type}: {p_value:.4f}")
        print(f"Log-rank test results for {cancer_type} saved as 'logrank_test_results_{cancer_type}.txt'.")

    kmf.fit(df['OS_MONTHS'], event_observed=df['OS_STATUS'])
    at_risk_data = kmf.event_table[['at_risk']]
    at_risk_data.to_csv('at_risk_data.csv', index=True)
    print("At risk data saved as 'at_risk_data.csv'.")
    
    c_index = cox.concordance_index_
    print(f'C-index: {c_index:.4f}')

    return cox



from tqdm import tqdm

import shap
import torch
import numpy as np
from tqdm import tqdm

import shap
import numpy as np
import torch
import shap
import numpy as np
import random
def compute_shap_values(cox_model, classification_features, test_features, excluded_columns=None, device="cuda", num_samples=200):
  
    classification_features = np.array(classification_features)
    test_features = np.array(test_features)

    background_data = shap.sample(classification_features, num_samples)
    explainer = shap.KernelExplainer(cox_model.predict_partial_hazard, background_data)

    print("Computing SHAP values...")
    try:
        shap_values = explainer.shap_values(test_features) 
    except Exception as e:
        print(f"Error while computing SHAP values: {e}")
        shap_values = []

    print(f"SHAP values shape: {shap_values.shape}")

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0]) 

    if len(shap_values.shape) == 1:
        print("SHAP values are 1D, expanding to match the number of features.")
        shap_values = np.repeat(shap_values, test_features.shape[1]).reshape(-1, test_features.shape[1])

    test_data_full = test_features 
    return shap_values, test_data_full

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def shap_analysis(shap_values,
                  test_data_full,
                  feature_names,
                  cancer_labels,
                  cancer_name_map,
                  excluded_columns=None,
                  save_prefix="shap"):
  
    for c, cname in cancer_name_map.items():
        mask     = (cancer_labels == c)
        sv_full  = shap_values[mask, :]
        data_sub = test_data_full[mask, :]

    
        local_excluded = [] if excluded_columns is None else list(excluded_columns)
    
        if 'uname' in feature_names:
            local_excluded.append(feature_names.index('uname'))
        
        if cname.lower() == "prostate.cancer":
            if 'Gender_F' in feature_names:
                local_excluded.append(feature_names.index('Gender_F'))
        if cname.lower() in ["prostate.cancer", "breast.cancer"]:
            if 'Gender_F' in feature_names:
                local_excluded.append(feature_names.index('Gender_F'))

        if local_excluded:
            sv_kept   = np.delete(sv_full, local_excluded, axis=1)
            data_kept = np.delete(data_sub, local_excluded, axis=1)
            feat_kept = [f for i, f in enumerate(feature_names) if i not in local_excluded]
        else:
            sv_kept   = sv_full
            data_kept = data_sub
            feat_kept = feature_names

        if sv_kept.ndim == 1:
            sv_kept = sv_kept.reshape(-1, 1)

        if sv_kept.shape[1] != data_kept.shape[1]:
            raise ValueError(
                f"Shape mismatch for class {cname}: shap has {sv_kept.shape[1]} features, "
                f"data has {data_kept.shape[1]}"
            )

     
        shap_df = pd.DataFrame(sv_kept, columns=feat_kept)
        shap_df.to_csv(f"{save_prefix}_raw_shap_{cname}.csv", index=False)
        vals_df = pd.DataFrame(data_kept, columns=feat_kept)
        vals_df.to_csv(f"{save_prefix}_raw_values_{cname}.csv", index=False)
        combined = pd.concat([
            shap_df.add_suffix("_shap"),
            vals_df.add_suffix("_value")
        ], axis=1)
        combined.to_csv(f"{save_prefix}_combined_{cname}.csv", index=False)

        shap.summary_plot(
            sv_kept,
            data_kept,
            feature_names=feat_kept,
            show=False
        )
        plt.title(f"SHAP summary – {cname}")
        plt.savefig(f"{save_prefix}_summaryos_{cname}.png", dpi=300)
        plt.close()

        # Save feature importance table
        mean_abs = np.abs(sv_kept).mean(axis=0)
        order    = np.argsort(-mean_abs)
        pd.DataFrame({
            "feature":       np.array(feat_kept)[order],
            "mean_abs_shap": mean_abs[order]
        }).to_csv(f"{save_prefix}_importanceos_{cname}.csv", index=False)

if __name__ == "__main__":
   
    data_train = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_os/ft_trainos.csv', sep=',', index_col=0)
    data_test = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_os/ft_testos.csv', sep=',', index_col=0)
    labels_train = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_os/labels_train.csv', sep=',', index_col=0).squeeze('columns')
    labels_test = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_os/labels_test.csv', sep=',', index_col=0).squeeze('columns')
    
    data_train['STAGE_HIGHEST_RECORDED'] = data_train['STAGE_HIGHEST_RECORDED'].fillna('Unknown').astype(str)
    data_test ['STAGE_HIGHEST_RECORDED'] = data_test ['STAGE_HIGHEST_RECORDED'].fillna('Unknown').astype(str)

    stage_ohe_train = pd.get_dummies(data_train['STAGE_HIGHEST_RECORDED'], prefix='Stage')
    stage_ohe_test  = pd.get_dummies(data_test ['STAGE_HIGHEST_RECORDED'], prefix='Stage')
    
 
    stage_ohe_train, stage_ohe_test = stage_ohe_train.align(
        stage_ohe_test,
        join='outer',  
        axis=1,        
        fill_value=0    
    )
    stage_ohe_train.columns = [col.replace("Stage_", "") for col in stage_ohe_train.columns]
    stage_ohe_test.columns  = [col.replace("Stage_", "") for col in stage_ohe_test.columns]
    
    data_train = pd.concat(
        [ data_train.drop(columns=['STAGE_HIGHEST_RECORDED']), stage_ohe_train ],
        axis=1
    )
    data_test  = pd.concat(
        [ data_test .drop(columns=['STAGE_HIGHEST_RECORDED']), stage_ohe_test  ],
        axis=1
    )
    
    genes = data_train.loc[:, 'ABL1':'YES1'].columns.tolist()
    ppi_df = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_os/string_interactions.tsv', sep='\t')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    train_features, test_features = process_data_with_global_attributes(data_train, data_test, ppi_df, genes, device)
    print("Train features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)
    
    n_features = train_features.shape[1]
    n_classes = len(set(labels_train))
    
    encoder = LabelEncoder()
    encoded_labels_train = encoder.fit_transform(labels_train)
    encoded_labels_test = encoder.transform(labels_test)
    
   
    label_encoder = LabelEncoder()
    cancer_types = label_encoder.fit_transform(labels_train) 
    print(f"Encoded cancer types: {cancer_types[:10]}")  
    
   
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32).to(device), torch.tensor(encoded_labels_train, dtype=torch.long).to(device))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32).to(device), torch.tensor(encoded_labels_test, dtype=torch.long).to(device))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
  
    model = FTTransformer(
        num_features=n_features,
        num_classes=n_classes,
        embed_dim=512,
        num_heads=2,
        ff_dim=768,
        num_layers=8,
        dropout_rate=0.1
    ).to(device)
  
    initialize_weights(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = FocalLoss(alpha=1, gamma=2)
    
    history = train_classification_model(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50,
        device=device
    )
    
    original_feature_names = data_train.columns.tolist()
    feature_name_map = {original_feature_names[i]: i for i in range(len(original_feature_names))}

   
    classification_features, test_features = process_data_with_global_attributes(
        data_train, data_test, ppi_df, genes, device
    )

    classification_features, test_features, feature_name_map = feature_selection_on_both(
        classification_features, test_features, data_train
    )
    selected_feature_names = list(feature_name_map.keys())

   
    os_months_train = data_train["OS_MONTHS"].values
    os_status_train = data_train["OS_STATUS"].values
    os_months_test  = data_test["OS_MONTHS"].values
    os_status_test  = data_test["OS_STATUS"].values

    os_train = data_train[['OS_MONTHS', 'OS_STATUS']]
    os_test  = data_test[['OS_MONTHS', 'OS_STATUS']]

    cancer_name_map = dict(zip(
        label_encoder.transform(label_encoder.classes_),
        label_encoder.classes_
    ))
        
    df_all = pd.DataFrame(classification_features, columns=selected_feature_names)
    df_all['Cancer_Type'] = cancer_types
    df_all['OS_MONTHS']   = os_months_train
    df_all['OS_STATUS']   = os_status_train

    df_all_test = pd.DataFrame(test_features, columns=selected_feature_names)
    df_all_test['Cancer_Type'] = encoded_labels_test
    df_all_test['OS_MONTHS']   = os_months_test
    df_all_test['OS_STATUS']   = os_status_test
   
    per_cancer_cox  = {}
    per_cancer_shap = {}
    

    prostate_code = None
    for code, name in cancer_name_map.items():
        if name == "Prostate.Cancer":
            prostate_code = code
            break
    if prostate_code is None:
        raise ValueError("在 cancer_name_map 中未找到 'Prostate.Cancer'。")
    
    c_name = "Prostate.Cancer"
    
    
    mask_train = (df_all['Cancer_Type'] == prostate_code)
    df_p_train = df_all.loc[mask_train, selected_feature_names + ['OS_MONTHS', 'OS_STATUS']].copy().dropna()
    
    mask_test = (df_all_test['Cancer_Type'] == prostate_code)
    df_p_test  = df_all_test.loc[mask_test,  selected_feature_names + ['OS_MONTHS', 'OS_STATUS']].copy().dropna()
    
    
    continuous_cols = []
    binary_cols     = []
    for col in selected_feature_names:
        if col in ['CN_Burden', 'LogSNV_Mb']:
            continuous_cols.append(col)
        else:
            binary_cols.append(col)
    
    # 对 Continuous 列做标准化
    scaler = StandardScaler()
    df_p_train[continuous_cols] = scaler.fit_transform(df_p_train[continuous_cols])
    df_p_test[continuous_cols]  = scaler.transform(df_p_test[continuous_cols])
    
    # 4) 删除 nunique<=1 的列
    drop_cols = [col for col in selected_feature_names if df_p_train[col].nunique() <= 1]
    if drop_cols:
        print(f"[{c_name}] 删除 nunique≤1 列：{drop_cols}")
    keep_cols = [c for c in selected_feature_names if c not in drop_cols]
    if len(keep_cols) == 0:
        raise RuntimeError(f"[{c_name}] 删除恒定列后没有可用特征！")
    
    df_p_train = df_p_train[keep_cols + ['OS_MONTHS', 'OS_STATUS']]
    df_p_test  = df_p_test[keep_cols + ['OS_MONTHS', 'OS_STATUS']]
    
    
    vt = VarianceThreshold(threshold=1e-3)
    vt.fit(df_p_train[keep_cols])
    mask_var = vt.get_support()
    keep_after_var = [keep_cols[i] for i, keep in enumerate(mask_var) if keep]
    if len(keep_after_var) < len(keep_cols):
        removed = set(keep_cols) - set(keep_after_var)
        print(f"[{c_name}] 方差阈值过滤后，删除列：{removed}")
    keep_cols = keep_after_var
    
    if len(keep_cols) == 0:
        raise RuntimeError(f"[{c_name}] 方差过滤后无可用特征！")
    df_p_train = df_p_train[keep_cols + ['OS_MONTHS', 'OS_STATUS']]
    df_p_test  = df_p_test[keep_cols + ['OS_MONTHS', 'OS_STATUS']]
    
    # 6) 高相关过滤（当任意两列 corr>0.9，就删掉后者那列）
    Xcorr = df_p_train[keep_cols].corr().abs()
    upper = Xcorr.where(np.triu(np.ones(Xcorr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high = [idx for idx in upper.index if upper.loc[idx, col] > 0.9]
        to_drop.update(high)
    if to_drop:
        print(f"[{c_name}] 删除高度相关列：{to_drop}")
        keep_cols = [c for c in keep_cols if c not in to_drop]
    
    if len(keep_cols) == 0:
        raise RuntimeError(f"[{c_name}] 高相关过滤后无特征！")
    df_p_train = df_p_train[keep_cols + ['OS_MONTHS', 'OS_STATUS']]
    df_p_test  = df_p_test[keep_cols + ['OS_MONTHS', 'OS_STATUS']]
    
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(df_p_train, duration_col='OS_MONTHS', event_col='OS_STATUS')
    print(f"[{c_name}] Cox 拟合完成，系数预览：")
    print(cox.summary)  
    
   
    n_bg = min(50, df_p_train.shape[0])
    background = shap.sample(df_p_train[keep_cols].to_numpy(), n_bg)
    
    explainer = shap.KernelExplainer(cox.predict_log_partial_hazard, background)
    X_test_np = df_p_test[keep_cols].to_numpy()
    
    print(f"[{c_name}] 开始计算 SHAP（样本数={X_test_np.shape[0]}）……")
    shap_vals = explainer.shap_values(X_test_np)
    
    
    plt.figure(figsize=(6,5))
    shap.summary_plot(
        shap_vals,
        X_test_np,
        feature_names=keep_cols,
        show=False
    )
    plt.title(f"SHAP summary – {c_name}")
    plt.savefig(f"shap_summary_{c_name}.png", dpi=300)
    plt.close()
    print(f"[{c_name}] 已保存 SHAP summary 图：shap_summary_{c_name}.png")
    
    mean_abs = np.abs(shap_vals).mean(axis=0)
    ranking = [(keep_cols[i], float(mean_abs[i])) for i in np.argsort(-mean_abs)]
    pd.DataFrame({
        "feature":       [f for f,_ in ranking],
        "mean_abs_shap": [v for _,v in ranking]
    }).to_csv(f"shap_importance_{c_name}.csv", index=False)
    print(f"[{c_name}] 已保存 SHAP 排名：shap_importance_{c_name}.csv")
