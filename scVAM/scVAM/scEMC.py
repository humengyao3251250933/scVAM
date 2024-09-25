import argparse
import os
import numpy as np
import torch
from time import time
import load_data as loader
from network import scEMC
from preprocess import read_dataset, normalize
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def parse_arguments(data_para):
    parser = argparse.ArgumentParser(description='scEMC')
    parser.add_argument('--n_clusters', default=data_para['K'], type=int)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('-el1', '--encodeLayer1', nargs='+', default=[256, 64, 32, 8])
    parser.add_argument('-el2', '--encodeLayer2', nargs='+', default=[256, 64, 32, 8])
    parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[24, 64, 256])
    parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[24, 20])
    parser.add_argument('--dataset', default=data_para)
    parser.add_argument("--view_dims", default=data_para['n_input'])
    parser.add_argument('--name', type=str, default=data_para[1])
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=500, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float, help='fuzziness of clustering loss')
    parser.add_argument('--phi1', default=0.001, type=float, help='pre coefficient of KL loss')
    parser.add_argument('--phi2', default=0.001, type=float, help='coefficient of KL loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='result/')
    parser.add_argument('--ae_weight_file', default='model.pth.tar')
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--embedding_file', action='store_true', default=False)
    parser.add_argument('--prediction_file', action='store_true', default=False)
    parser.add_argument('--sigma1', default=2.5, type=float)
    parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--f1', default=2000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lam1', default=1, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    return parser.parse_args()

def prepare_data(dataset, size_factors=True, normalize_input=True, logtrans_input=True):
    """Prepare the data for the model."""
    data = sc.AnnData(np.array(dataset))
    data = read_dataset(data, transpose=False, test_split=False, copy=True)
    data = normalize(data, size_factors=size_factors, normalize_input=normalize_input, logtrans_input=logtrans_input)
    return data

# def plot_tsne(features, labels, save_path, perplexity=50, learning_rate=600):
#     # label_names = ['CD14 Mono', 'CD16 Mono', 'CD4 Memory', 'CD4 Naive', 'CD56 bright NK','CD8 Effector_1','CD8 Effector_2','CD8 Memory_1','CD8 Memory_2','CD8 Naive','cDC2','gdT','GMP','HSC','LMPP','MAIT','Memory B','Naive B','NK','pDC','Plasmablast','Prog_B 1','Prog_B 2','Prog_DC','Prog_MK','Prog_RBC','Treg']
#     label_names = ["CD4 T cells", "CD8 T cells", "B cells", "NK cells", "Monocytes",
#     "Dendritic cells", "Plasmacytoid dendritic cells", "Megakaryocytes",
#     "Erythrocytes", "Granulocytes", "T helper cells", "Regulatory T cells",
#     "Effector T cells", "Memory T cells", "Naive T cells", "Gamma-delta T cells"]
#     tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
#     tsne_results = tsne.fit_transform(features)
#     plt.figure(figsize=(10, 10))
#     unique_labels = np.unique(labels)
#     colors = plt.cm.get_cmap('tab20', len(unique_labels))
#
#     for i, label in enumerate(unique_labels):
#         indices = np.where(labels == label)
#         if label >= len(label_names):
#             print(f"Label index {label} out of range for label_names with length {len(label_names)}")
#             continue
#
#         if np.max(indices) >= tsne_results.shape[0]:
#             print(f"Index {np.max(indices)} out of range for tsne_results with shape {tsne_results.shape}")
#             continue
#
#         plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], s=5, color=colors(i), label=label_names[label])
#     plt.xlabel('tSNE_1')
#     plt.ylabel('tSNE_2')
#     plt.legend(loc='best', bbox_to_anchor=(1, 1))
#     plt.savefig(save_path, bbox_inches='tight', dpi=600)
#     plt.show()
#
# def main():
#     my_data_dic = loader.ALL_data
#     for i_d in my_data_dic:
#         data_para = my_data_dic[i_d]
#     args = parse_arguments(data_para)
#     # mat_file_path = os.path.join(loader.path, args.dataset[1] + ".mat")
#     # if not os.path.exists(mat_file_path):
#     #     print(f"文件 {mat_file_path} 不存在，请检查文件路径。")
#     #     return
#     X, Y = loader.load_data(args.dataset)
#     labels = Y[0].copy().astype(np.int32)
#     # Prepare data
#     adata1 = prepare_data(X[0])
#     adata2 = prepare_data(X[1], size_factors=False, normalize_input=False, logtrans_input=False)
#     y = labels
#     input_size1 = adata1.n_vars
#     input_size2 = adata2.n_vars
#
#     encodeLayer1 = list(map(int, args.encodeLayer1))
#     encodeLayer2 = list(map(int, args.encodeLayer2))
#     decodeLayer1 = list(map(int, args.decodeLayer1))
#     decodeLayer2 = list(map(int, args.decodeLayer2))
#
#     model = scEMC(input_dim1=input_size1, input_dim2=input_size2, tau=args.tau,
#                   encodeLayer1=encodeLayer1,
#                   encodeLayer2=encodeLayer2,
#                   decodeLayer1=decodeLayer1, decodeLayer2=decodeLayer2,
#                   activation='elu', sigma1=args.sigma1, sigma2=args.sigma2, gamma=args.gamma,
#                   cutoff=args.cutoff, phi1=args.phi1, phi2=args.phi2, device=args.device).to(args.device)
#
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
#
#     t0 = time()
#     if args.ae_weights is None:
#         model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
#                                    X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors,
#                                    batch_size=args.batch_size,
#                                    epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
#     else:
#         if os.path.isfile(args.ae_weights):
#             print("==> loading checkpoint '{}'".format(args.ae_weights))
#             checkpoint = torch.load(args.ae_weights)
#             model.load_state_dict(checkpoint['ae_state_dict'])
#         else:
#             print("==> no checkpoint found at '{}'".format(args.ae_weights))
#             raise ValueError
#
#     print('Pretraining time: %d seconds.' % int(time() - t0))
#
#     # get k
#     latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))
#     latent = latent.cpu().numpy()
#
#     # Save the extracted features (jump embeddings)
#     feature_path = os.path.join(args.save_dir, str(args.run) + "_features.csv")
#     np.savetxt(feature_path, latent, delimiter=",")
#
#     print('Feature extraction completed.')
#     print('Total time: %d seconds.' % int(time() - t0))
#
#     # t-SNE and plot
#     plot_tsne(latent, y, os.path.join(args.save_dir, str(args.run) + "_tsne.png"))
#
# if __name__ == "__main__":
#     main()
def plot_tsne_subplots(features, labels, save_path, params_list):
    """
    Plot t-SNE subplots with different parameters and save them in one image.

    Args:
        features (numpy.ndarray): The feature matrix.
        labels (numpy.ndarray): The labels corresponding to the features.
        save_path (str): The path to save the combined plot.
        params_list (list of dict): A list of dictionaries, each containing parameters for t-SNE.
    """
    # label_names = ["CD4 T cells", "CD8 T cells", "B cells", "NK cells", "Monocytes",
    #                "Dendritic cells", "Plasmacytoid dendritic cells", "Megakaryocytes",
    #                "Erythrocytes", "Granulocytes", "T helper cells", "Regulatory T cells",
    #                "Effector T cells", "Memory T cells", "Naive T cells", "Gamma-delta T cells"]
    label_names = ['CD14 Mono', 'CD16 Mono', 'CD4 Memory', 'CD4 Naive', 'CD56 bright NK', 'CD8 Effector_1',
                   'CD8 Effector_2', 'CD8 Memory_1', 'CD8 Memory_2', 'CD8 Naive', 'cDC2', 'gdT', 'GMP', 'HSC', 'LMPP',
                   'MAIT', 'Memory B', 'Naive B', 'NK', 'pDC', 'Plasmablast', 'Prog_B 1', 'Prog_B 2', 'Prog_DC',
                   'Prog_MK', 'Prog_RBC', 'Treg']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    subplot_labels = ['A', 'B', 'C']

    for ax, params, subplot_label in zip(axes.flat, params_list + [None], subplot_labels + [None]):
        if params is None:
            ax.axis('off')
            continue

        tsne = TSNE(n_components=2, perplexity=params['perplexity'], learning_rate=params['learning_rate'],
                    random_state=42)
        tsne_results = tsne.fit_transform(features)

        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))

        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)[0]
            if label >= len(label_names):
                print(f"Label index {label} out of range for label_names with length {len(label_names)}")
                continue

            if np.max(indices) >= tsne_results.shape[0]:
                print(f"Index {np.max(indices)} out of range for tsne_results with shape {tsne_results.shape}")
                continue

            ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1], s=5, color=colors(i),
                       label=label_names[label])
        ax.set_xlabel('tSNE_1')
        ax.set_ylabel('tSNE_2')
        ax.text(-0.1, 1.05, subplot_label, transform=ax.transAxes, size=20, weight='bold')

    # Add legend to the last subplot
    handles, labels = axes.flat[-2].get_legend_handles_labels()
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(handles, labels, loc='center', title="Cell Types")

    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()


def main():
    my_data_dic = loader.ALL_data
    for i_d in my_data_dic:
        data_para = my_data_dic[i_d]
        args = parse_arguments(data_para)

        X, Y = loader.load_data(args.dataset)
        labels = Y[0].copy().astype(np.int32)

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

        t0 = time()
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

        print('Pretraining time: %d seconds.' % int(time() - t0))

        latent = model.encodeBatch(torch.tensor(adata1.X).to(args.device), torch.tensor(adata2.X).to(args.device))
        latent = latent.cpu().numpy()

        feature_path = os.path.join(args.save_dir, str(args.run) + "_features.csv")
        np.savetxt(feature_path, latent, delimiter=",")

        print('Feature extraction completed.')
        print('Total time: %d seconds.' % int(time() - t0))

        params_list = [
            {'perplexity': 30, 'learning_rate': 200},
            {'perplexity': 50, 'learning_rate': 600},
            {'perplexity': 100, 'learning_rate': 1000}
        ]
        plot_tsne_subplots(latent, y, os.path.join(args.save_dir, str(args.run) + "_tsne_subplots.png"), params_list)


if __name__ == "__main__":
    main()