# model_testing.py
```python
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from data_preprocessing import load_and_preprocess_data, create_sequences, build_graph
from model_definition import MultiScaleGATv2_LSTM

# -------------------------------
# Configuration
# -------------------------------
config = {
    'dynamic_data_path': '/path/to/dynamic.csv',
    'static_data_path': '/path/to/static.csv',
    'grid_data_path': '/path/to/grid.csv',
    'sequence_length': 24,
    'forecast_horizons': [1, 6, 24],
    'model_state_path': '/path/to/best_model.pth',
    'output_dir': '/path/to/output'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load and preprocess data
    _, _, test_data, static_data, grid_df, features_to_scale, target_scaler = \
        load_and_preprocess_data(config)

    # Build graph (for node mapping consistency)
    _ , node_mapping = build_graph(static_data, pd.DataFrame(), grid_df)

    # Create test sequences
    test_sequences, test_targets, test_nodes = create_sequences(
        test_data, features_to_scale, config['sequence_length'], config['forecast_horizons'], node_mapping
    )
    test_dataset = TensorDataset(test_sequences, test_targets, test_nodes)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    node_feature_dim = len(node_mapping)
    sequence_feature_dim = test_sequences.size(2)
    edge_dim = 0  # Not needed at test time since embeddings precomputed
    forecast_horizon = test_targets.size(1)

    model = MultiScaleGATv2_LSTM(
        node_feature_dim, sequence_feature_dim,  # dummy node_feature_dim ignored if embeddings stored
        gat_out_channels=0, gat_heads=0, lstm_hidden_dim=0,
        lstm_layers=0, edge_dim=edge_dim, forecast_horizon=forecast_horizon
    ).to(device)
    model.load_state_dict(torch.load(config['model_state_path'], map_location=device))
    model.eval()

    # Evaluate
    start_test = time.time()
    all_preds, all_tgts, all_nodes = [], [], []
    with torch.no_grad():
        for seq, tgt, nd in test_loader:
            seq, tgt, nd = seq.to(device), tgt.to(device), nd.to(device)
            out = model(seq, nd)
            all_preds.append(out.cpu().numpy())
            all_tgts.append(tgt.cpu().numpy())
            all_nodes.append(nd.cpu().numpy())
    test_time = time.time() - start_test
    print(f"Total Test Time: {test_time:.2f} seconds")

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_tgts = np.concatenate(all_tgts, axis=0)
    all_nodes = np.concatenate(all_nodes, axis=0)

    # Inverse transform
    preds_inv = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    tgts_inv  = target_scaler.inverse_transform(all_tgts.reshape(-1, 1)).reshape(all_tgts.shape)

    # Prepare results DataFrame
    records = []
    num_h = preds_inv.shape[1]
    for i in range(len(all_nodes)):
        for h in range(num_h):
            records.append({
                'Node': int(all_nodes[i]),
                'Horizon': h+1,
                'Actual': float(tgts_inv[i, h]),
                'Predicted': float(preds_inv[i, h])
            })
    results_df = pd.DataFrame.from_records(records)
    results_df.to_csv(config['output_csv_path'], index=False)
    print(f"Saved results CSV to {config['output_csv_path']}")

    # Compute metrics per horizon
    metrics = {}
    for h in range(num_h):
        act = results_df[results_df['Horizon'] == h+1]['Actual']
        pred = results_df[results_df['Horizon'] == h+1]['Predicted']
        metrics[f'Horizon_{h+1}'] = {
            'MAE': mean_absolute_error(act, pred),
            'RMSE': np.sqrt(mean_squared_error(act, pred)),
            'MAPE': (np.mean(np.abs((act - pred) / np.clip(act, 1e-8, None))) * 100),
            'R2': r2_score(act, pred),
            'Correlation': pearsonr(act, pred)[0]
        }
    print("Evaluation metrics:", metrics)
