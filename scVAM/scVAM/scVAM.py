import itertools
import os
from datetime import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import load_data as loader
from network import scEMC
from preprocess import read_dataset, normalize
from scEMC import parse_arguments, prepare_data
from utils import *
import time

def train_and_evaluate(args, data_para):
    X, Y = loader.load_data(args.dataset)
    labels = Y[0].copy().astype(np.int32)

    # 准备数据
    adata1 = prepare_data(X[0])
    adata2 = prepare_data(X[1], size_factors=False, normalize_input=False, logtrans_input=False)
    y = labels
    input_size1 = adata1.n_vars
    input_size2 = adata2.n_vars

    encodeLayer1 = list(map(int, args.encodeLayer1))
    encodeLayer2 = list(map(int, args.encodeLayer2))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    decodeLayer2 = list(map(int, args.decodeLayer2))

    model = scEMC(input_dim1=input_size1, input_dim2=input_size2, tau=args.tau,
                  encodeLayer1=encodeLayer1,
                  encodeLayer2=encodeLayer2,
                  decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
                  activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
                  cutoff=args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    t0 = time.time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                   X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors,
                                   batch_size=args.batch_size,
                                   epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time.time() - t0))

    latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))
    latent = latent.cpu().numpy()
    if args.n_clusters == -1:
        n_clusters = GetCluster(latent, res=args.resolution, n=args.n_neighbors)
    else:
        print("n_cluster is defined as " + str(args.n_clusters))
        n_clusters = args.n_clusters

    y_pred, _, _, _ = model.fit(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                                X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors, y=y,
                                n_clusters=n_clusters, batch_size=args.batch_size, num_epochs=args.maxiter,
                                update_interval=args.update_interval, tol=args.tol, lr=args.lr,
                                save_dir=args.save_dir, lam1=args.lam1, lam2=args.lam2)
    print('Total time: %d seconds.' % int(time.time() - t0))
    # 调试输出
    print(f'Unique labels in y: {np.unique(y)}')
    print(f'Unique labels in y_pred: {np.unique(y_pred)}')
    print(f'Shape of y: {y.shape}')
    print(f'Shape of y_pred: {y_pred.shape}')
    y_pred_ = best_map(y, y_pred)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 4)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 4)
    print('Final: ARI= %.4f, NMI= %.4f' % (ari, nmi))

    return nmi, ari


def grid_search(param_grid, data_para):
    best_nmi = 0
    best_ari = 0
    best_params = None

    param_combinations = list(itertools.product(*param_grid.values()))

    results = {'params': [], 'nmi': [], 'ari': []}

    for param_combination in param_combinations:
        args = parse_arguments(data_para)

        for param_name, param_value in zip(param_grid.keys(), param_combination):
            setattr(args, param_name, param_value)

        nmi, ari = train_and_evaluate(args, data_para)

        results['params'].append(param_combination)
        results['nmi'].append(nmi)
        results['ari'].append(ari)

        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari
            best_params = param_combination

    return best_nmi, best_ari, best_params, results


def plot_heatmap(results, param_grid, param1, param2, score='nmi', filename=None):
    param1_values = param_grid[param1]
    param2_values = param_grid[param2]

    heatmap_data = np.zeros((len(param1_values), len(param2_values)))

    for i, param1_value in enumerate(param1_values):
        for j, param2_value in enumerate(param2_values):
            for k, params in enumerate(results['params']):
                if params[list(param_grid.keys()).index(param1)] == param1_value and params[list(param_grid.keys()).index(param2)] == param2_value:
                    heatmap_data[i, j] = results[score][k]

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, xticklabels=param2_values, yticklabels=param1_values, cmap='viridis')
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f'{score.upper()} Scores Heatmap for {param1} and {param2}')

    if filename:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    else:
        plt.show()


def main():
    my_data_dic = loader.ALL_data
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]

    param_grid = {
        'lr': [1e-3, 1e-4],
        'batch_size': [32, 64, 128, 256],
        'maxiter': [500, 1000],
        'gamma': [0.1, 0.2],
        'sigma1': [2.5, 3.0],
        'sigma2': [1.5, 2.0],
        'tau': [1.0, 1.5],
    }

    best_nmi, best_ari, best_params, results = grid_search(param_grid, data_para)
    print(f'Best NMI: {best_nmi}, Best ARI: {best_ari}, Best Params: {best_params}')

    plot_heatmap(results, param_grid, 'lr', 'batch_size', score='nmi', filename='nmi_heatmap.png')
    plot_heatmap(results, param_grid, 'lr', 'batch_size', score='ari', filename='ari_heatmap.png')


if __name__ == "__main__":
    main()
