import os
import sys
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, precision_score, recall_score, f1_score

from utils import CV_PATIENTS, CHECKPOINT_DIR, setup_dirs
from dataset import SienaSTFTDataset
from model import SeizureCNN2D

# Hyperparameters
NUM_FOLDS      = 5
BATCH_SIZE     = 64
EPOCHS         = 30
EARLY_STOP_PAT = 5
POS_WEIGHT     = 10.0
DECISION_THRESHOLD = 0.5
MIN_IMPROVEMENT = 0.01 # Require at least 1% improvement for early stopping counter reset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, threshold=DECISION_THRESHOLD):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.sigmoid(model(X_batch.to(device)))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels).astype(int)

    if len(np.unique(all_labels)) < 2:
        return 0.0

    roc_auc = roc_auc_score(all_labels, all_probs)
    pr, rc, _ = precision_recall_curve(all_labels, all_probs)
    
    print(f"\n  ROC-AUC: {roc_auc*100:.2f}%  |  AUCPR: {auc(rc, pr)*100:.2f}%")
    print(f"  {'Threshold':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")

    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (all_probs >= t).astype(int)
        marker = ' ←' if abs(t - threshold) < 1e-5 else ''
        print(f"  {t:>10.1f} {recall_score(all_labels, preds, zero_division=0)*100:>9.2f}% "
              f"{precision_score(all_labels, preds, zero_division=0)*100:>9.2f}% "
              f"{f1_score(all_labels, preds, zero_division=0)*100:>9.2f}%{marker}")

    return recall_score(all_labels, (all_probs >= threshold).astype(int), zero_division=0)

class FoldTrainer:
    """Encapsulates training state to cleanly handle interrupts without globals."""
    def __init__(self, fold, train_loader, eval_loader):
        self.fold = fold
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = SeizureCNN2D().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
        
        self.best_recall = 0.0
        self.current_epoch = 0
        self.last_ckpt = os.path.join(CHECKPOINT_DIR, f'last_fold{fold+1}.pth')
        self.best_ckpt = os.path.join(CHECKPOINT_DIR, f'best_recall_fold{fold+1}.pth')
        
        signal.signal(signal.SIGINT, self._interrupt_handler)

    def _interrupt_handler(self, sig, frame):
        print(f"\nInterrupt received — saving Fold {self.fold+1} state...")
        self.save_checkpoint(self.last_ckpt)
        sys.exit(0)

    def save_checkpoint(self, path):
        torch.save({
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_recall': self.best_recall,
        }, path)

    def train(self):
        start_epoch, patience = 0, 0

        if os.path.exists(self.last_ckpt):
            ckpt = torch.load(self.last_ckpt, map_location=device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch, self.best_recall = ckpt['epoch'] + 1, ckpt['best_recall']
            print(f"Resumed Fold {self.fold+1} at epoch {start_epoch} | Best recall: {self.best_recall*100:.2f}%")

        for epoch in range(start_epoch, EPOCHS):
            self.current_epoch = epoch
            train_loss = train_epoch(self.model, self.train_loader, self.optimizer, self.criterion)
            print(f"Fold {self.fold+1} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}")

            recall = evaluate(self.model, self.eval_loader, DECISION_THRESHOLD)
            self.scheduler.step(recall)

            if recall > self.best_recall + MIN_IMPROVEMENT:
                self.best_recall = recall
                patience = 0
                self.save_checkpoint(self.best_ckpt)
                print(f"  ★ New best recall: {self.best_recall*100:.2f}%")
            else:
                patience += 1
                print(f"  No significant improvement — patience {patience}/{EARLY_STOP_PAT}")
            
            self.save_checkpoint(self.last_ckpt)

            if patience >= EARLY_STOP_PAT:
                print(f"  Early stopping at epoch {epoch+1}.")
                break
            print()
            
        # Clear signal handler when done
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        return self.best_recall


if __name__ == '__main__':

    setup_dirs()

    print(f"Device: {device}")
    cv_dataset = SienaSTFTDataset(CV_PATIENTS.tolist())
    
    if len(cv_dataset) == 0:
        print("No data loaded. Check ROOT_DIR and cache.")
        sys.exit(1)

    gkf = GroupKFold(n_splits=NUM_FOLDS)
    fold_recalls = []

    for fold, (train_idx, eval_idx) in enumerate(gkf.split(cv_dataset.X, cv_dataset.y, groups=cv_dataset.groups)):
        print(f"{'═'*60}\nFOLD {fold+1} / {NUM_FOLDS}\n{'═'*60}")
        
        train_loader = DataLoader(Subset(cv_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        eval_loader  = DataLoader(Subset(cv_dataset, eval_idx), batch_size=BATCH_SIZE*4, shuffle=False, pin_memory=True)

        trainer = FoldTrainer(fold, train_loader, eval_loader)
        best_recall = trainer.train()
        fold_recalls.append(best_recall)
        
    print(f"{'═'*60}\nCROSS-VALIDATION SUMMARY\n{'═'*60}")
    for i, r in enumerate(fold_recalls):
        print(f"  Fold {i+1}: {r*100:.2f}%")
    print(f"\n  Mean recall: {np.mean(fold_recalls)*100:.2f}% (+/- {np.std(fold_recalls)*100:.2f}%)")