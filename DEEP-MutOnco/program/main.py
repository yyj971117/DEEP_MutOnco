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
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, trial_number=None, patience=100, test_loader=None):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': [], 'auc': [], 'aupr': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train, total_train = 0, 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(output, dim=1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        train_acc = correct_train / total_train if total_train > 0 else 0.0
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        all_probs = []
        all_true_labels = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                val_loss += criterion(output, y_batch).item()
                _, predicted = torch.max(output, dim=1)
                correct_val += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)

                # Save true labels and predicted probabilities for AUC and AUPR calculation
                all_true_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(F.softmax(output, dim=1).cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val if total_val > 0 else 0.0
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Calculate AUC and AUPR
        try:
            auc = roc_auc_score(all_true_labels, all_probs, multi_class='ovr')
        except ValueError:
            auc = 0.0

        try:
            aupr = average_precision_score(all_true_labels, all_probs, average='macro')
        except ValueError:
            aupr = 0.0
        
        history['auc'].append(auc)
        history['aupr'].append(aupr)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"AUC: {auc:.4f}, AUPR: {aupr:.4f}")

        # Correct usage: using f-string formatting
        plt.plot(history['train_loss'], label=f'Train Loss = {history["train_loss"][-1]:.4f}', linestyle='-', color='blue')
        plt.plot(history['val_loss'], label=f'Validation Loss = {history["val_loss"][-1]:.4f}', linestyle='-', color='red')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig('loss_curve1.png', dpi=300)
        plt.close()

        plt.plot(history['train_acc'], label=f'Train Accuracy = {history["train_acc"][-1]:.4f}', linestyle='-', color='green')
        plt.plot(history['val_acc'], label=f'Validation Accuracy = {history["val_acc"][-1]:.4f}', linestyle='-', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.savefig('accuracy_curve1.png', dpi=300)
        plt.close()

        plt.plot(history['auc'], label=f'AUC = {history["auc"][-1]:.4f}', linestyle='-', color='purple')
        plt.plot(history['aupr'], label=f'AUPR = {history["aupr"][-1]:.4f}', linestyle='-', color='pink')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('AUC and AUPR over Epochs')
        plt.savefig('auc_aupr_curve.png', dpi=300)
        plt.close()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if trial_number is not None:
                torch.save(model.state_dict(), f"best_model_trial{trial_number}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if test_loader is not None:
        model.eval()
        true_labels_list = []
        predicted_probs_list = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                probs = F.softmax(output, dim=1).cpu().numpy()
                predicted_probs_list.append(probs)
                true_labels_list.extend(y_batch.cpu().numpy())

        predicted_probs_array = np.vstack(predicted_probs_list)  # shape = [n_samples, n_classes]
        true_labels_array = np.array(true_labels_list)  # shape = [n_samples,]

        try:
            auc_test = roc_auc_score(true_labels_array, predicted_probs_array, multi_class='ovr')
        except ValueError:
            auc_test = None

        try:
            aupr_test = average_precision_score(true_labels_array, predicted_probs_array, average='macro')
        except ValueError:
            aupr_test = None

        print(f"Final AUC on Test Set: {auc_test:.4f}")
        print(f"Final AUPR on Test Set: {aupr_test:.4f}")
        

    return history


import optuna
import numpy as np

class MockTrial:
    def __init__(self, params, trial_number=None):
        self._params = params
        self.number = trial_number

    def suggest_categorical(self, name, choices):
        return self._params.get(name)

    def suggest_int(self, name, low, high, step=1):
        return self._params.get(name)

    def suggest_float(self, name, low, high, step=0.1, log=False):
        value = self._params.get(name)
        if log:
            # If log=True, transform to log scale
            return np.exp(value)
        else:
            return value

# Example of setting trial number
manual_params = {
    'embed_dim': 512,
    'num_heads': 2,
    'ff_dim': 768,
    'dropout_rate': 0.27257709513888054,
    'lr': np.log(1e-5)  # manually passing log-scaled value
}

# Instantiate MockTrial with a trial number
manual_trial = MockTrial(manual_params, trial_number=0)
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
def objective(trial):
    n_classes = 38
    import time
    start_time = time.time()

    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
    num_heads_choices = [h for h in [1, 2, 4, 8, 16, 32, 64, 128] if embed_dim % h == 0]
    if not num_heads_choices:
        raise optuna.exceptions.TrialPruned()
    num_heads = trial.suggest_categorical("num_heads", num_heads_choices)
    ff_dim = trial.suggest_int("ff_dim", 256, 1024, step=256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_train = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/ft_train_with_new_features.csv', sep=',', index_col=0)
    data_test = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/ft_test_with_new_features.csv', sep=',', index_col=0)
    labels_train = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/labels_train.csv', sep=',', index_col=0).squeeze('columns')
    labels_test = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/labels_test.csv', sep=',', index_col=0).squeeze('columns')

    genes = data_train.loc[:, 'ABL1':'YES1'].columns.tolist()
    ppi_df = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/string_interactions.tsv', sep='\t')

    train_features, test_features = process_data_with_global_attributes(data_train, data_test, ppi_df, genes, device)
    print("Train features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)

    original_feature_names = data_train.columns.tolist()
    ppi_feature_names = genes
    feature_names = original_feature_names + ppi_feature_names
    print("len(feature_names) =", len(feature_names))

    n_features = train_features.shape[1]
    n_types = len(set(labels_train))

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoded_labels_train = encoder.fit_transform(labels_train)
    encoded_labels_test = encoder.transform(labels_test)

    n_splits = 5
    fold_results = []
    fold_results_file = "fold_results_trial.csv"

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_splits, random_state=0)
    train_index_list, val_index_list = [], []
    for train_index, val_index in sss.split(train_features, encoded_labels_train):
        train_index_list.append(train_index)
        val_index_list.append(val_index)

    for fold in range(n_splits):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        fold_train_x = torch.tensor(train_features[train_index_list[fold]], dtype=torch.float32)
        fold_val_x = torch.tensor(train_features[val_index_list[fold]], dtype=torch.float32)
        fold_train_y = torch.tensor(encoded_labels_train[train_index_list[fold]], dtype=torch.long)
        fold_val_y = torch.tensor(encoded_labels_train[val_index_list[fold]], dtype=torch.long)

        x_test = torch.tensor(test_features, dtype=torch.float32)
        y_test = torch.tensor(encoded_labels_test, dtype=torch.long)

        train_dataset = TensorDataset(fold_train_x, fold_train_y)
        val_dataset = TensorDataset(fold_val_x, fold_val_y)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = FTTransformer(
            num_features=n_features,
            num_classes=n_types,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=8,
            dropout_rate=dropout_rate
        ).to(device)

        initialize_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = FocalLoss(alpha=1, gamma=2)

        history = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=150,
            device=device,
            trial_number=trial.number,
            patience=100
        )

        
        model.eval()
        true_labels_list = []
        predicted_probs_list = []
        predicted_labels_list = []

        with torch.no_grad():
            test_dataset_new = TensorDataset(x_test, y_test)
            test_loader_new = DataLoader(test_dataset_new, batch_size=32, shuffle=False)
            for x_batch, y_batch in test_loader_new:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                probs = F.softmax(output, dim=1).cpu().numpy()
                _, predicted = torch.max(output, dim=1)
                predicted_labels_list.extend(predicted.cpu().numpy())
                true_labels_list.extend(y_batch.cpu().numpy())
                predicted_probs_list.append(probs)

        
        predicted_probs_array = np.vstack(predicted_probs_list)  # shape = [n_samples, n_classes]
        true_labels_array = np.array(true_labels_list)  # shape = [n_samples,]

        acc = accuracy_score(true_labels_array, predicted_labels_list)
        precision = precision_score(true_labels_array, predicted_labels_list, average='weighted', zero_division=1)
        recall = recall_score(true_labels_array, predicted_labels_list, average='weighted', zero_division=1)
        f1 = f1_score(true_labels_array, predicted_labels_list, average='weighted')
        
        try:
            auc_val = roc_auc_score(true_labels_array, predicted_probs_array, multi_class='ovr')
        except ValueError:
            auc_val = None
        try:
            aupr_val = average_precision_score(true_labels_array, predicted_probs_array, average='macro')
        except ValueError:
            aupr_val = None

        fold_result = {
            'trial_number': trial.number,
            'fold': fold + 1,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_val if auc_val is not None else 0.0,
            'aupr': aupr_val if aupr_val is not None else 0.0
        }
        fold_results.append(fold_result)
        
      
        fold_results_file = "fold_results_trial.csv"
        with open(fold_results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if os.stat(fold_results_file).st_size == 0:
                writer.writerow(['Trial Number', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'AUPR'])
            writer.writerow([
                fold_result['trial_number'],
                fold_result['fold'],
                fold_result['accuracy'],
                fold_result['precision'],
                fold_result['recall'],
                fold_result['f1'],
                fold_result['auc'],
                fold_result['aupr']
            ])

        fpr_all, tpr_all, roc_auc_all = {}, {}, {}
        for i in range(n_classes):
            fpr_all[i], tpr_all[i], _ = roc_curve(true_labels_array == i, predicted_probs_array[:, i])
            roc_auc_all[i] = auc(fpr_all[i], tpr_all[i])

        plt.figure(figsize=(16, 12))
        for i in range(n_classes):
            plt.plot(fpr_all[i], tpr_all[i], label=f'{encoder.classes_[i]}: AUC = {roc_auc_all[i]:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Overall Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig('overall_roc_curve.png', dpi=300)
        plt.close()


        precision_all, recall_all, pr_auc_all = {}, {}, {}
        for i in range(n_classes):
            precision_all[i], recall_all[i], _ = precision_recall_curve(true_labels_array == i, predicted_probs_array[:, i])
            pr_auc_all[i] = auc(recall_all[i], precision_all[i])

        plt.figure(figsize=(16, 12))
        for i in range(n_classes):
            plt.plot(recall_all[i], precision_all[i], label=f'{encoder.classes_[i]}: AUPR = {pr_auc_all[i]:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Overall Precision-Recall Curve')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig('overall_aupr_curve.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(16, 12))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(true_labels_array == i, predicted_probs_array[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{encoder.classes_[i]}: AUC = {roc_auc:.2f}')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Each Class')
        plt.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.savefig('test_auc_curve.png', dpi=300)
        plt.close()

        plt.figure(figsize=(16, 12))
        for i in range(n_classes):
            precision_i, recall_i, _ = precision_recall_curve(true_labels_array == i, predicted_probs_array[:, i])
            aupr = auc(recall_i, precision_i)
            plt.plot(recall_i, precision_i, label=f'{encoder.classes_[i]}: AUPR = {aupr:.2f}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Each Class')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout() 
        plt.savefig('test_aupr_curve.png', dpi=300)
        plt.close()
        
        cm = confusion_matrix(true_labels_array, predicted_labels_list)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.close()
        
        
        plt.figure(figsize=(16, 12)) 
        for i in range(n_classes):
            precision_i, recall_i, _ = precision_recall_curve(true_labels_array == i, predicted_probs_array[:, i])
            
            max_precision_idx = np.argmax(precision_i)
            max_recall_idx = np.argmax(recall_i)
            
            plt.plot(recall_i, precision_i, label=f'{encoder.classes_[i]}: Precision = {precision_i[max_precision_idx]:.2f}, Recall = {recall_i[max_recall_idx]:.2f}')
            
            plt.annotate(f'{precision_i[max_precision_idx]:.2f}, {recall_i[max_recall_idx]:.2f}',
                         (recall_i[max_recall_idx], precision_i[max_precision_idx]),
                         textcoords="offset points", xytext=(0, 10), ha='center')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Each Class')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()  
        plt.savefig('precision_recall_curve.png', dpi=300)
        plt.close()


        
        fpr, tpr, _ = roc_curve(true_labels_array == 1, predicted_probs_array[:, 1])
        roc_auc_val = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(f'roc_curve_fold_{fold+1}.png')
        plt.close()

    print("\nCross-validation results:")
    best_fold = max(fold_results, key=lambda x: x['accuracy'])
    for result in fold_results:
        print(f"Fold {result['fold']}: Accuracy: {result['accuracy']:.4f}, "
              f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, "
              f"F1: {result['f1']:.4f}, AUC: {result['auc']:.4f}, AUPR: {result['aupr']:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    results_file = "trials_results.csv"
    if not os.path.exists(results_file):
        with open(results_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Trial Number", "Embed Dim", "Num Heads", "FF Dim", "Dropout Rate", "Learning Rate", "Best Accuracy", "Elapsed Time"])

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial.number, embed_dim, num_heads, ff_dim, dropout_rate, lr, best_fold['accuracy'], elapsed_time])

    print(f"Total execution time for trial {trial.number}: {elapsed_time:.2f} seconds")
    return best_fold['accuracy']

objective(manual_trial)


if __name__ == "__main__":
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)

    best_params = study.best_params
    embed_dim = best_params["embed_dim"]
    num_heads = best_params["num_heads"]
    ff_dim = best_params["ff_dim"]
    dropout_rate = best_params["dropout_rate"]
    lr = best_params["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    data_train = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/ft_train_with_new_features.csv', sep=',', index_col=0)
    data_test = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/ft_test_with_new_features.csv', sep=',', index_col=0)
    labels_train = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/labels_train.csv', sep=',', index_col=0).squeeze('columns')
    labels_test = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/labels_test.csv', sep=',', index_col=0).squeeze('columns')

    
    genes = data_train.loc[:, 'ABL1':'YES1'].columns.tolist()
    ppi_df = pd.read_csv('~/DEEP_MutOnco/DEEP-MutOnco/dataset/data_class/string_interactions.tsv', sep='\t')

    
    train_features, test_features = process_data_with_global_attributes(
        data_train, data_test, ppi_df, genes, device
    )
    print("Train features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)

    
    original_feature_names = data_train.columns.tolist()
    ppi_feature_names = genes
    feature_names = original_feature_names + ppi_feature_names
    print("len(feature_names) =", len(feature_names))

    n_features = train_features.shape[1]
    n_types = len(set(labels_train))

    
    encoder = LabelEncoder()
    encoded_labels_train = encoder.fit_transform(labels_train)
    encoded_labels_test = encoder.transform(labels_test)

    
    x_train_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(encoded_labels_train, dtype=torch.long).to(device)
    x_test_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(encoded_labels_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    model = FTTransformer(
        num_features=n_features,
        num_classes=n_types,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=8,
        dropout_rate=dropout_rate
    ).to(device)

    initialize_weights(model)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=1, gamma=2)

    history = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=150,
        device=device,
        trial_number="best_model",
        patience=100,
        test_loader=test_loader  
    )

    
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, dim=1)
            true_labels.extend(y_batch.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            predicted_probs.extend(probs.cpu().numpy())

    final_acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nBest model final test accuracy: {final_acc:.4f}")

