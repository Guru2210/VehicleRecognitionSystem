"""tsne_map.py

Usage examples (PowerShell):
python .\tsne_map.py --model resnet50d --n-samples 2000 --perplexity 30 --out tsne_resnet50d.png

The script loads vectors from pickles/<model>/vectors.pkl and optional metadata pickles/jsondata.pkl,
computes PCA to reduce to 50 dims (optional) and then t-SNE to 2D, saves embeddings and a PNG scatter.
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter


def load_vectors(model_name, vectors_path=None):
    if vectors_path is None:
        vectors_path = os.path.join('pickles', model_name, 'vectors.pkl')
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")
    with open(vectors_path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors


def load_jsondata():
    path = os.path.join('pickles', 'jsondata.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_matrix(vectors, n_samples=None, random_state=42):
    keys = list(vectors.keys())
    if n_samples is not None and n_samples < len(keys):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(keys), size=n_samples, replace=False)
        keys = [keys[i] for i in idx]
    mat = np.array([vectors[k] for k in keys])
    return keys, mat


def pick_colors(labels):
    # Map labels to colors. For many labels, use a hash-based color.
    unique = list(dict.fromkeys(labels))
    n = len(unique)
    if n <= 20:
        cmap = plt.get_cmap('tab20')
        mapping = {lab: cmap(i % 20) for i, lab in enumerate(unique)}
    else:
        # use hsv from label hash
        mapping = {}
        for lab in unique:
            h = (hash(lab) % 360) / 360.0
            mapping[lab] = plt.cm.hsv(h)
    colors = [mapping[l] for l in labels]
    return colors, mapping


def main():
    parser = argparse.ArgumentParser(description='Compute t-SNE map from saved vectors')
    parser.add_argument('--model', '-m', default='resnet50d', help='model name used in pickles folder')
    parser.add_argument('--vectors', help='optional path to vectors.pkl (overrides model)')
    parser.add_argument('--n-samples', type=int, default=2000, help='number of samples to embed (default 2000, use None for all)')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--learning-rate', type=float, default=200.0, help='t-SNE learning rate')
    parser.add_argument('--pca-dim', type=int, default=50, help='PCA dimension before t-SNE (0 to skip)')
    parser.add_argument('--out', '-o', default=None, help='output PNG file path (default: pickles/<model>/tsne.png)')
    parser.add_argument('--emb-out', default=None, help='output pickle for embeddings (default: pickles/<model>/tsne_embeddings.pkl)')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    model = args.model
    vectors_path = args.vectors if args.vectors else None
    vectors = load_vectors(model, vectors_path)
    jsondata = load_jsondata()

    n_samples = args.n_samples
    if n_samples is not None and n_samples <= 0:
        n_samples = None

    keys, mat = prepare_matrix(vectors, n_samples=n_samples, random_state=args.random_state)
    print(f"Loaded {mat.shape[0]} vectors, dim={mat.shape[1]}")

    X = mat
    if args.pca_dim and args.pca_dim > 0 and X.shape[1] > args.pca_dim:
        print(f"Running PCA -> {args.pca_dim} dims")
        pca = PCA(n_components=args.pca_dim, random_state=args.random_state)
        X = pca.fit_transform(X)

    print(f"Running t-SNE (perplexity={args.perplexity}, lr={args.learning_rate})")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=args.learning_rate, init='pca', random_state=args.random_state)
    X2 = tsne.fit_transform(X)

    # Labels for coloring: if jsondata exists and is list-of-dicts, map by resized_name
    labels = None
    if jsondata is not None:
        # support both list-of-dicts or DataFrame-like
        try:
            if isinstance(jsondata, list):
                mapping = {e['resized_name']: e.get('label', 'Unknown') for e in jsondata}
            else:
                # DataFrame-like
                df = jsondata
                if hasattr(df, 'set_index') and 'resized_name' in getattr(df, 'columns', []):
                    mapping = dict(zip(df['resized_name'].tolist(), df.get('label', ['Unknown']*len(df))))
                else:
                    mapping = {}
            labels = [mapping.get(k, 'Unknown') for k in keys]
        except Exception:
            labels = ['Unknown'] * len(keys)
    else:
        labels = ['Unknown'] * len(keys)

    colors, mapping = pick_colors(labels)

    out_dir = os.path.join('pickles', model)
    os.makedirs(out_dir, exist_ok=True)
    out_png = args.out if args.out else os.path.join(out_dir, f'tsne_{model}.png')
    emb_out = args.emb_out if args.emb_out else os.path.join(out_dir, 'tsne_embeddings.pkl')

    print(f"Saving embeddings to {emb_out} and plot to {out_png}")
    with open(emb_out, 'wb') as f:
        pickle.dump({'keys': keys, 'embeddings': X2, 'labels': labels}, f)

    plt.figure(figsize=(10, 10))
    xs = X2[:, 0]
    ys = X2[:, 1]
    plt.scatter(xs, ys, c=colors, s=6, alpha=0.8)
    plt.title(f"t-SNE map ({model}) n={len(keys)}")

    # create legend for up to 20 classes
    unique = list(dict.fromkeys(labels))
    if len(unique) <= 20:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=mapping[lab], label=lab) for lab in unique]
        plt.legend(handles=patches, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print('Done')


if __name__ == '__main__':
    main()
