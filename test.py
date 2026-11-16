import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report

from train import NUM_FOLDS
from utils import TEST_PATIENTS, CHECKPOINT_DIR
from dataset import SienaSTFTDataset
from model import SeizureCNN2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5 # Change based on train.py sweep results

def run_test():
    print(f"Loading TEST data for patients: {TEST_PATIENTS}")
    test_dataset = SienaSTFTDataset(TEST_PATIENTS)
    
    if len(test_dataset) == 0:
        print("No test data found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    # Load the best fold model (Scan all folds, choose one with highest recall)
    best_ckpt_path  = None
    best_ckpt_recall = -1.0

    for fold in range(1, NUM_FOLDS + 1):
        path = os.path.join(CHECKPOINT_DIR, f'best_recall_fold{fold}.pth')
        if os.path.exists(path):
            ckpt   = torch.load(path, map_location='cpu')
            recall = ckpt.get('best_recall', 0.0)
            if recall > best_ckpt_recall:
                best_ckpt_recall = recall
                best_ckpt_path   = path

    print(f"Using checkpoint: {best_ckpt_path} (recall: {best_ckpt_recall*100:.2f}%)")
    
    if not os.path.exists(best_ckpt_path):
        print(f"Checkpoint not found: {best_ckpt_path}")
        return

    model = SeizureCNN2D().to(device)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device)['model'])
    model.eval()

    all_probs = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = torch.sigmoid(model(X_batch.to(device)))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels).astype(int)
    preds = (all_probs >= THRESHOLD).astype(int)

    # 1. Base Metrics
    print(f"\n{'═'*40}\nTEST SET RESULTS (Threshold: {THRESHOLD})\n{'═'*40}")
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs)*100:.2f}%\n")
    print(classification_report(all_labels, preds, target_names=["Non-Seizure", "Seizure"]))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Seizure", "Seizure"])
    disp.plot(cmap='Blues')
    plt.title(f'Test Set Confusion Matrix (Threshold {THRESHOLD})')
    plt.savefig('test_confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix to 'test_confusion_matrix.png'")

    # 3. Probability Distribution Plot
    plt.figure(figsize=(10, 5))
    plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.5, label='Non-Seizure (0)', density=True)
    plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.5, label='Seizure (1)', density=True, color='red')
    plt.axvline(THRESHOLD, color='k', linestyle='dashed', linewidth=1, label=f'Threshold ({THRESHOLD})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.savefig('test_prob_distribution.png')
    plt.close()
    print("Saved probability distribution to 'test_prob_distribution.png'")

if __name__ == '__main__':
    run_test()