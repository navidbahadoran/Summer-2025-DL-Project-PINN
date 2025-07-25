# gnn/data/graph_dataset.py

import pandas as pd
import torch
import geopandas as gpd
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import config
from shapely.geometry import Point


def load_edges_from_geometry(shapefile_path, node_coords, simplify=True):
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(epsg=3857)
    if simplify:
        gdf['geometry'] = gdf['geometry'].simplify(1000)

    node_points = gpd.GeoSeries([Point(xy) for xy in node_coords], crs="EPSG:4326").to_crs(epsg=3857)
    spatial_index = gdf.sindex
    node_to_poly = []
    for pt in node_points:
        match = list(spatial_index.intersection(pt.bounds))
        matched_index = None
        for idx in match:
            if gdf.geometry.iloc[idx].contains(pt):
                matched_index = idx
                break
        node_to_poly.append(matched_index)

    valid_idx_map = {}
    filtered_nodes = []
    for i, poly_id in enumerate(node_to_poly):
        if poly_id is not None:
            if poly_id not in valid_idx_map:
                valid_idx_map[poly_id] = len(valid_idx_map)
            filtered_nodes.append(i)
    if len(filtered_nodes) < len(node_coords):
        print(f"[INFO] Filtering {len(node_coords) - len(filtered_nodes)} unmatched nodes.")

    poly_to_nodes = {}
    for node_id, poly_id in enumerate(node_to_poly):
        if poly_id is not None:
            mapped_id = valid_idx_map[poly_id]
            poly_to_nodes.setdefault(mapped_id, []).append(node_id)

    poly_id_to_valid = {k: v for v, k in enumerate(valid_idx_map.keys())}

    edges = []
    for poly_id in valid_idx_map.keys():
        geom = gdf.geometry.iloc[poly_id]
        neighbors = gdf.geometry.touches(geom)
        neighbor_ids = gdf.index[neighbors].intersection(valid_idx_map.keys())
        for neighbor_id in neighbor_ids:
            i_nodes = poly_to_nodes[valid_idx_map[poly_id]]
            j_nodes = poly_to_nodes[valid_idx_map[neighbor_id]]
            for ni in i_nodes:
                for nj in j_nodes:
                    if ni != nj:
                        edges.append((ni, nj))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_static_graph(data_csv_path, shapefile_path, history=3):
    print("[INFO] Read CSV file from",data_csv_path)
    df = pd.read_csv(data_csv_path)
    df = df.sort_values(by=["lon", "lat", "t"]).reset_index(drop=True)

    # Clean floating point keys and define unique nodes
    df['lon'] = df['lon'].round(6)
    df['lat'] = df['lat'].round(6)
    df['pos'] = list(zip(df['lon'], df['lat']))
    node_positions = df['pos'].drop_duplicates().tolist()
    node_idx_map = {pos: i for i, pos in enumerate(node_positions)}
    df['node_id'] = df['pos'].map(node_idx_map)

    n_nodes = len(node_positions)
    t_max = int(df['t'].max())

    features_per_time = []
    targets_per_time = []

    for t in range(history, t_max):
        past_ts = list(range(t - history, t))
        df_t = df[df['t'].isin(past_ts)]
        df_y = df[df['t'] == t]

        if df_t.shape[0] != n_nodes * history:
            continue

        # Build X matrix with shape (n_nodes, history)
        x = np.zeros((n_nodes, history), dtype=np.float32)
        valid = True
        for node in range(n_nodes):
            u_vals = df_t[df_t['node_id'] == node].sort_values('t')['u'].values
            if len(u_vals) != history:
                valid = False
                print(f"[SKIP] t={t}: node {node} missing history")
                break
            x[node] = u_vals
        if not valid:
            continue

        y = df_y.sort_values('node_id')['u'].values.astype(np.float32)
        features_per_time.append(torch.tensor(x, dtype=torch.float32))
        targets_per_time.append(torch.tensor(y, dtype=torch.float32))

    if not targets_per_time:
        raise RuntimeError("No valid data samples were found.")

    # Fit and apply target scaler
    all_targets = torch.cat(targets_per_time).numpy().reshape(-1, 1)
    scaler_y = StandardScaler().fit(all_targets)
    targets_per_time = [
        torch.tensor(scaler_y.transform(y.reshape(-1, 1)).flatten(), dtype=torch.float32)
        for y in targets_per_time
    ]
    torch.save(scaler_y, config["gnn_scaler_y"])

    # === Scale X features as well ===
    x_all = torch.cat(features_per_time).numpy()
    scaler_x = StandardScaler().fit(x_all)
    features_per_time = [
        torch.tensor(scaler_x.transform(x.numpy()), dtype=torch.float32)
        for x in features_per_time
    ]
    torch.save(scaler_x, config["gnn_scaler_x"])


    print("[INFO] Matching counties by geometry...")
    edge_index = load_edges_from_geometry(shapefile_path, node_positions)
    max_node_id = edge_index.max().item()
    if max_node_id >= n_nodes:
        raise ValueError(f"[ERROR] Invalid edge index: max ID = {max_node_id}, but only {n_nodes} nodes.")

    data_list = []
    for i, (x, y) in enumerate(zip(features_per_time, targets_per_time)):

        data = Data(x=x, y=y, edge_index=edge_index.clone())
        data_list.append(data)
    print(f"[INFO] Final valid graph samples: {len(data_list)}")
    torch.save((data_list, node_positions), config["gnn_dataset_path"])
    print("[INFO] Graph dataset has been saved in: ", config["gnn_dataset_path"])
    return data_list, node_positions 



if __name__ == "__main__":
    csv_path = config["data_path"]
    shapefile_path = config["shape_file_path"]
    dataset, coords = build_static_graph(csv_path, shapefile_path)
    print(f"Generated {len(dataset)} graph snapshots with {dataset[0].num_nodes} nodes")
