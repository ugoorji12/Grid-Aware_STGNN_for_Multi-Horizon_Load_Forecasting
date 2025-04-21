import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx

def load_and_preprocess_data(config):
    """Load and preprocess datasets."""
    dynamic_data = pd.read_csv(config['dynamic_data_path'])
    static_data = pd.read_csv(config['static_data_path'])
    grid_df = pd.read_csv(config['grid_data_path'])

    # Convert datetime column and sort
    dynamic_data['datetime'] = pd.to_datetime(dynamic_data['date_time'])
    dynamic_data = dynamic_data.sort_values(by=['state', 'datetime'])

    # Clip negative renewable values
    for col in ['pv_value', 'on_wind_value', 'off_wind_value']:
        dynamic_data[col] = dynamic_data[col].clip(lower=0)

    # Create target by shifting load backward and drop NaNs.
    dynamic_data['target_consumption'] = dynamic_data.groupby('state')['load'].shift(-1)
    dynamic_data.dropna(subset=['target_consumption'], inplace=True)

    # Split into train, validation, test
    train_data = dynamic_data[(dynamic_data['datetime'] >= config['train_start_date']) &
                              (dynamic_data['datetime'] < config['train_end_date'])].copy()
    val_data = dynamic_data[(dynamic_data['datetime'] >= config['val_start_date']) &
                            (dynamic_data['datetime'] < config['val_end_date'])].copy()
    test_data = dynamic_data[(dynamic_data['datetime'] >= config['test_start_date']) &
                             (dynamic_data['datetime'] < config['test_end_date'])].copy()

    # Create time index if needed
    for data in [train_data, val_data, test_data]:
        data['time_index'] = ((data['datetime'] - data['datetime'].min()).dt.total_seconds() // 3600).astype(int)

    # Define features to scale (exclude target)
    features_to_scale = [
         'load', 'pv_value', 'on_wind_value', 'off_wind_value',
         'TOTAL HOURLY RAIN (mm)_mean', 'TOTAL HOURLY RAIN (mm)_std',
         'ATMOSPHERIC PRESSURE AT STATION LEVEL (mB)_mean',
         'ATMOSPHERIC PRESSURE AT STATION LEVEL (mB)_std',
         'GLOBAL RADIATION (KJ/mÂ²)_mean', 'GLOBAL RADIATION (KJ/mÂ²)_std',
         'AIR TEMPERATURE (Â°C)_mean', 'AIR TEMPERATURE (Â°C)_std',
         'DEW POINT TEMPERATURE (Â°C)_mean', 'DEW POINT TEMPERATURE (Â°C)_std',
         'MAXIMUM TEMPERATURE FOR THE LAST HOUR (Â°C)_mean',
         'MAXIMUM TEMPERATURE FOR THE LAST HOUR (Â°C)_std',
         'MINIMUM TEMPERATURE FOR THE LAST HOUR (Â°C)_mean',
         'MINIMUM TEMPERATURE FOR THE LAST HOUR (Â°C)_std',
         'REL HUMIDITY FOR THE LAST HOUR (%)_mean',
         'REL HUMIDITY FOR THE LAST HOUR (%)_std', 'WIND DIRECTION (gr)_mean',
         'WIND DIRECTION (gr)_std', 'WIND MAXIMUM GUST (m/s)_mean',
         'WIND MAXIMUM GUST (m/s)_std', 'WIND SPEED (m/s)_mean',
         'WIND SPEED (m/s)_std', 'year', 'month', 'day', 'hour', 'dayofweek',
         'weekofyear', 'quarter', 'is_holiday', 'season', 'GDP (R$ billion)',
         'GDP per capita (R$)', 'Population (millions)', 'plant_cap'
    ]

    # Scale dynamic features
    feature_scaler = RobustScaler()
    train_data[features_to_scale] = feature_scaler.fit_transform(train_data[features_to_scale])
    val_data[features_to_scale] = feature_scaler.transform(val_data[features_to_scale])
    test_data[features_to_scale] = feature_scaler.transform(test_data[features_to_scale])

    # Scale target separately
    target_scaler = RobustScaler()
    train_data['target_consumption'] = target_scaler.fit_transform(train_data[['target_consumption']])
    val_data['target_consumption'] = target_scaler.transform(val_data[['target_consumption']])
    test_data['target_consumption'] = target_scaler.transform(test_data[['target_consumption']])

    return train_data, val_data, test_data, static_data, grid_df, features_to_scale, target_scaler


def build_graph(static_data, dynamic_data, grid_df):
    """Build graph from static and dynamic data."""
    G = nx.Graph()

    # Add nodes from static data.
    for idx, row in static_data.iterrows():
        state = row['state']
        G.add_node(state, **row.to_dict())

    # Attach time series data (if needed) from dynamic_data.
    for state, group in dynamic_data.groupby('state'):
        if state in G.nodes:
            G.nodes[state]['time_series'] = group.to_dict(orient='records')
        else:
            logging.warning(f"{state} found in dynamic data but not in static data.")

    # Add edges using grid_df.
    for idx, row in grid_df.iterrows():
        source = row['Source']
        target = row['Target']
        if source in G.nodes and target in G.nodes:
            if G.has_edge(source, target):
                G[source][target]['capacity'] += row.get('capacity', 0)
            else:
                G.add_edge(source, target, **row.to_dict())
        else:
            logging.warning(f"Edge ({source}, {target}) not added due to missing node.")

    # Create a mapping from state to index.
    node_mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
    pos_list, node_features_list = [], []
    for node in sorted(G.nodes()):
        data = G.nodes[node]
        x_coord = data.get('x', 0)
        y_coord = data.get('y', 0)
        pos_list.append((x_coord, y_coord))
        features = [x_coord, y_coord,
                    data.get('pv_pot', 0),
                    data.get('onw_pot', 0),
                    data.get('ofw_pot', 0)]
        node_features_list.append(features)

    # Normalize node features.
    node_features_df = pd.DataFrame(node_features_list, columns=['x', 'y', 'pv_pot', 'onw_pot', 'ofw_pot'])
    scaler = RobustScaler()
    normalized_node_features = scaler.fit_transform(node_features_df)
    node_features_tensor = torch.tensor(normalized_node_features, dtype=torch.float)
    pos_tensor = torch.tensor(pos_list, dtype=torch.float)

    # Build edge index and edge attributes.
    edge_index_list, edge_attr_list = [], []
    for source, target, data in G.edges(data=True):
        src_idx = node_mapping[source]
        tgt_idx = node_mapping[target]
        edge_index_list.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
        edge_attr = [
            data.get('capacity', 0),
            data.get('line_eff', 0),
            data.get('line_len', 0),
            data.get('line_carrier', 0)  # Ensure this is numeric.
        ]
        edge_attr_list.extend([edge_attr, edge_attr])

    edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr_df = pd.DataFrame(edge_attr_list, columns=['capacity', 'line_eff', 'line_len', 'line_carrier'])
    edge_scaler = RobustScaler()
    scaled_edge_attrs = edge_scaler.fit_transform(edge_attr_df)
    edge_attr_tensor = torch.tensor(scaled_edge_attrs, dtype=torch.float)

    graph_data = Data(x=node_features_tensor, edge_index=edge_index_tensor,
                      edge_attr=edge_attr_tensor, pos=pos_tensor)
    logging.info(f"Graph constructed with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges.")

    return graph_data, node_mapping, node_features_tensor, edge_index_tensor, edge_attr_tensor

def create_sequences(data, features_to_scale, sequence_length, horizons, node_mapping, return_hour=False):
    """Create sequences of inputs and targets."""
    sequences, targets, nodes = [], [], []
    hours = [] if return_hour else None
    max_horizon = max(horizons)

    grouped = data.groupby('state')
    for state, group in grouped:
        group = group.sort_values(by='datetime')
        values = group[features_to_scale].values
        target_values = group['target_consumption'].values
        state_idx = node_mapping.get(state)
        if state_idx is None:
            continue
        if len(group) >= sequence_length + max_horizon - 1:
            for i in range(len(group) - sequence_length - max_horizon + 2):
                seq = values[i:i+sequence_length]
                tgt = [target_values[i + sequence_length + h - 2] for h in horizons]
                sequences.append(seq)
                targets.append(tgt)
                nodes.append(state_idx)
                if return_hour:
                    forecast_hour = pd.to_datetime(group.iloc[i+sequence_length]['datetime']).hour
                    hours.append(forecast_hour)
        else:
            logging.warning(f"Not enough data for state {state} for the required sequence length.")

    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.float)
    nodes_tensor = torch.tensor(np.array(nodes), dtype=torch.long)

    if return_hour:
        hours_tensor = torch.tensor(np.array(hours), dtype=torch.long)
        return sequences_tensor, targets_tensor, nodes_tensor, hours_tensor
    return sequences_tensor, targets_tensor, nodes_tensor

if __name__ == "__main__":
    # Example config
    config = {
        'dynamic_data_path': '/path/to/dynamic.csv',
        'static_data_path': '/path/to/static.csv',
        'grid_data_path': '/path/to/grid.csv',
        'train_start_date': '2017-01-01',
        'train_end_date': '2018-12-31',
        'val_start_date': '2019-01-01',
        'val_end_date': '2019-06-30',
        'test_start_date': '2019-07-01',
        'test_end_date': '2019-12-31'
    }

    # Load and preprocess data
    train_data, val_data, test_data, static_data, grid_df, features_to_scale, target_scaler = load_and_preprocess_data(config)
    graph_data, node_mapping, node_features_tensor, edge_index_tensor, edge_attr_tensor = build_graph(static_data, train_data, grid_df)
    train_sequences, train_targets, train_nodes = create_sequences(train_data, features_to_scale, 24, [1, 6, 24], node_mapping)
    val_sequences, val_targets, val_nodes = create_sequences(val_data, features_to_scale, 24, [1, 6, 24], node_mapping)
    test_sequences, test_targets, test_nodes, test_hours = create_sequences(test_data, features_to_scale, 24, [1, 6, 24], node_mapping, return_hour=True)
    
    # Save preprocessed data if needed
    torch.save(train_sequences, 'train_sequences.pt')
    torch.save(train_targets, 'train_targets.pt')
