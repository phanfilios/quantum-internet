#1/usr/bin/env python3
"""
qmicns_V0_0_1_QISKIT.PY
sicp V0.0.0.01 eeg CORE + Qiskit integration 
Usage example:
 python qmicns_V0.0.0.1_eeg_qiskit.py --csv
eeg_data.csv --model rf --outdir outputs_q
 PYTHON qmicns_V0.0.0.0.1_eeg_qiskit.py --csv
eeg_data.csv --model torch --use_cuda
-- n_qubits 6
NOTES.
- CSV expected columns:
timestamp,subject_id,brain-region,signal_ampl
itude,frequency_band , experimental_condition
 - Set n_qubits <= number of bands (default 5 bands -> 5 qubits lol)
"""

import argparse
import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StandardScaler , LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


try:
    from qiskit import QuantumCircuit, Aer
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, Operator
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False


BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

# -------------------------
# Utilities: load CSV & aggregate per sample (subject, region, condition)
# -------------------------

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    required = {"timestamp", "subject_id", "brain_region", "signal_amplitude", "frequency_band", "experimental_condition"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV misses required columns. Required: {required}. Found: {set(df.columns)}")
    df['frequency_band'] = df['frequency_band'].astype(str).str.capitalize()
    # aggregate
    agg = defaultdict(lambda: {b: [] for b in BANDS})
    for _, row in df.iterrows():
        key = (str(row['subject_id']), str(row['brain_region']), str(row['experimental_condition']))
        band = str(row['frequency_band']).capitalize()
        amp = float(row['signal_amplitude'])
        if band in BANDS:
            agg[key][band].append(amp)
    rows = []
    for key, band_dict in agg.items():
        subject, region, cond = key
        feat = {'subject_id': subject, 'brain_region': region, 'experimental_condition': cond}
        for b in BANDS:
            arr = np.array(band_dict[b]) if len(band_dict[b])>0 else np.array([0.0])
            feat[f'{b}_mean'] = float(np.mean(arr))
            feat[f'{b}_std'] = float(np.std(arr))
            feat[f'{b}_median'] = float(np.median(arr))
            feat[f'{b}_count'] = int(len(arr))
        rows.append(feat)
    features_df = pd.DataFrame(rows)
    return df, features_df

# -------------------------
# Qiskit: apply rotations mapped from band means -> returns expectations list
# -------------------------

def bands_to_qubit_angles(band_means, alpha=1.5, scale=0.01):
    # band_means: list in order BANDS
    angles = []
    for m in band_means:
        theta = alpha * np.tanh(scale * float(m))  # maps magnitude -> bounded angle
        angles.append(float(theta))
    return angles

def apply_q_ops_and_get_expectations(angles, n_qubits, backend_name='aer_simulator_statevector'):
    """
    angles: list of angles (one per qubit/band). n_qubits defines space (if angles shorter, rest identity)
    returns: list of expectation <Z> per qubit (length n_qubits)
    """
    if not HAVE_QISKIT:
        return [0.0]*n_qubits
    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(angles[:n_qubits]):
        qc.ry(theta, i)
  
    sv = Statevector.from_instruction(qc)  
    dm = DensityMatrix(sv)  
    expectations = []
   
    for q in range(n_qubits):
        reduced = partial_trace(dm, [i for i in range(n_qubits) if i != q])
        rho = reduced.data
        expZ = float(np.real(rho[0,0] - rho[1,1]))
        expectations.append(expZ)
    return expectations

# -------------------------
# Visualization helpers
# -------------------------
  
def visualize_basic(features_df, outdir):
    os.makedirs(outdir, exist_ok=True)
    melt_mean = features_df.melt(
        id_vars=['subject_id','brain_region','experimental_condition'],
        value_vars=[f'{b}_mean' for b in BANDS],
        var_name='band', value_name='mean_amplitude'
    )
    melt_mean['band'] = melt_mean['band'].str.replace('_mean','')
    plt.figure(figsize=(12,6))
    sns.boxplot(data=melt_mean, x='brain_region', y='mean_amplitude', hue='band')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p1 = os.path.join(outdir, 'band_mean_by_region.png')
    plt.savefig(p1, dpi=200); plt.close()
    print("Saved:", p1)

# -------------------------
# ML helpers: build matrices and train
# -------------------------
  
def build_ml_matrix_with_quantum(features_df, n_qubits, alpha, scale):
   
    feature_cols = []
    for b in BANDS:
        feature_cols += [f'{b}_mean', f'{b}_std', f'{b}_median', f'{b}_count']
    X_base = features_df[feature_cols].values.astype(np.float32)
    extra_feats = []
    for idx, row in features_df.iterrows():
        band_means = [row[f'{b}_mean'] for b in BANDS]
        angles = bands_to_qubit_angles(band_means, alpha=alpha, scale=scale)
        exps = apply_q_ops_and_get_expectations(angles, n_qubits)
        extra_feats.append(exps)
    extra_feats = np.array(extra_feats, dtype=np.float32)  

    X = np.hstack([X_base, extra_feats])
    y = features_df['experimental_condition'].values.astype(str)
    meta = features_df[['subject_id','brain_region']].copy()
    return X, y, meta, feature_cols + [f'Qexp_{i}' for i in range(n_qubits)]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32], n_classes=2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h)); layers.append(nn.ReLU()); layers.append(nn.BatchNorm1d(h))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train_torch_model(X_train, y_train, X_val, y_val, class_names, epochs=60, lr=1e-3, device='cpu'):
    le = LabelEncoder(); y_train_enc = le.fit_transform(y_train); y_val_enc = le.transform(y_val)
    input_dim = X_train.shape[1]; n_classes = len(class_names)
    model = SimpleMLP(input_dim, hidden_dims=[128,64], n_classes=n_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device); Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train_enc, dtype=torch.long).to(device); yv = torch.tensor(y_val_enc, dtype=torch.long).to(device)
    for ep in range(epochs):
        model.train(); opt.zero_grad(); logits = model(Xtr); loss = crit(logits, ytr); loss.backward(); opt.step()
        if ep % 10 == 0 or ep==epochs-1:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv)
                preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                acc = (preds == y_val_enc).mean()
            print(f"Epoch {ep:03d} loss={loss.item():.4f} val_acc={acc:.3f}")
    return model, le

def train_rf(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# -------------------------
# Main pipeline
# -------------------------
  
def main(args):
    if not HAVE_QISKIT:
        print("Warning: Qiskit not available. Quantum expectations will be zeros. Install qiskit/qiskit-aer for full behavior.")
    if args.model == 'torch' and not HAVE_TORCH:
        print("Torch not installed. Falling back to RF.")
        args.model = 'rf'

    print("Loading CSV:", args.csv)
    df, features_df = load_and_prepare(args.csv)
    print("Prepared feature rows:", len(features_df))
    os.makedirs(args.outdir, exist_ok=True)

    print("Basic visualization...")
    visualize_basic(features_df, args.outdir)

    print("Building ML matrix with quantum expectations (n_qubits=%d)..." % args.n_qubits)
    X, y, meta, all_feature_cols = build_ml_matrix_with_quantum(features_df, args.n_qubits, args.alpha, args.scale)
    print("Final feature dim:", X.shape[1])

    le_cond = LabelEncoder(); y_enc = le_cond.fit_transform(y); class_names = list(le_cond.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

    model_path = os.path.join(args.outdir, f'model_{args.model}_q{args.n_qubits}.pkl')
    if args.model == 'torch':
        device = 'cuda' if (args.use_cuda and torch.cuda.is_available()) else 'cpu'
        print("Training Torch MLP on device:", device)
        model, label_encoder = train_torch_model(X_train_s, y_train, X_test_s, y_test, class_names, epochs=args.epochs, device=device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'label_encoder_classes': label_encoder.classes_.tolist(),
            'feature_cols': all_feature_cols
        }, model_path)
        print("Saved torch model to", model_path)
      
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test_s, dtype=torch.float32).to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        print(classification_report(y_test, label_encoder.inverse_transform(preds)))
    else:
        print("Training RandomForest baseline...")
        rf = train_rf(X_train_s, y_train)
        joblib.dump({'model': rf, 'scaler': scaler, 'label_encoder_classes': le_cond.classes_, 'feature_cols': all_feature_cols}, model_path)
        print("Saved RF model to", model_path)
        preds = rf.predict(X_test_s)
        print(classification_report(y_test, preds))
        cm = confusion_matrix(y_test, preds, labels=le_cond.classes_)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_cond.classes_, yticklabels=le_cond.classes_, cmap='Blues')
        plt.title('Confusion matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'confusion_matrix.png'))
        plt.close()
        print("Saved confusion matrix.")

    features_df.to_csv(os.path.join(args.outdir, 'features_table_with_q.csv'), index=False)
    print("Saved features to", os.path.join(args.outdir, 'features_table_with_q.csv'))
    print("Completed. Outputs in:", args.outdir)

# -------------------------
# CLI
# -------------------------
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QMICNS v3.3 EEG + Qiskit integration")
    parser.add_argument('--csv', type=str, required=True, help='Path to EEG CSV')
    parser.add_argument('--outdir', type=str, default='outputs_q', help='Output directory')
    parser.add_argument('--model', type=str, choices=['rf','torch'], default='rf', help='Model to train')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test fraction')
    parser.add_argument('--epochs', type=int, default=60, help='Epochs for torch')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--n_qubits', type=int, default=5, help='Number of qubits (<= number of bands recommended)')
    parser.add_argument('--alpha', type=float, default=1.5, help='Gain mapping band mean -> rotation angle')
    parser.add_argument('--scale', type=float, default=0.01, help='Scale used inside tanh for band means')
    args = parser.parse_args()
    main(args)
