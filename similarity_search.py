import argparse
import pickle
import os
import numpy as np
from PIL import Image

from img2vec import Img2Vec


def load_vectors(model_name):
    path = os.path.join('pickles', model_name, 'vectors.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectors pickle not found: {path}. Run vectorization first.")
    with open(path, 'rb') as f:
        vectors = pickle.load(f)
    return vectors


def load_jsondata():
    path = os.path.join('pickles', 'jsondata.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_similarity(query_vec, vectors):
    # vectors: dict filename -> vector (numpy array)
    keys = list(vectors.keys())
    mat = np.array([vectors[k] for k in keys])
    # ensure shapes
    q = np.array(query_vec)
    # compute cosine similarity
    dot = mat.dot(q)
    mat_norms = np.linalg.norm(mat, axis=1)
    q_norm = np.linalg.norm(q)
    sims = dot / (mat_norms * q_norm + 1e-10)
    return keys, sims


def main():
    parser = argparse.ArgumentParser(description='Query similarity search using finetuned model')
    parser.add_argument('--image', '-i', required=True, help='Path to query image (file)')
    parser.add_argument('--model', '-m', default='resnet50d', help='Model name used for vectors (default resnet50d)')
    parser.add_argument('--fine-tuned', '-f', default=None, help='Path to fine-tuned state_dict (.pth) to load')
    parser.add_argument('--topk', '-k', type=int, default=10, help='Top-K results to return')
    parser.add_argument('--resize', type=int, default=224, help='Resize shortest side to this size (default 224)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print('Query image not found:', args.image)
        return

    vectors = load_vectors(args.model)
    jsondata = load_jsondata()

    device_cuda = args.cuda
    img2vec = Img2Vec(cuda=device_cuda, model=args.model, fine_tuned_path=args.fine_tuned)

    # load query image
    img = Image.open(args.image).convert('RGB')
    if args.resize:
        img = img.resize((args.resize, args.resize))

    try:
        qvec = img2vec.get_vec(img, normalize_vec=True)
    except RuntimeError:
        qvec = img2vec.get_vec(img.convert('RGB'), normalize_vec=True)

    keys, sims = compute_similarity(qvec, vectors)

    # sort descending
    order = np.argsort(-sims)
    topk = min(args.topk, len(keys))
    print(f"Top {topk} similar images for {args.image} (model={args.model}):")
    for i in range(topk):
        idx = order[i]
        name = keys[idx]
        score = float(sims[idx])
        label = None
        if jsondata:
            # map resized_name to label
            mapping = {entry['resized_name']: entry['label'] for entry in jsondata}
            label = mapping.get(name)
        if label:
            print(f"{i+1}. {name}  (label: {label})  score: {score:.4f}")
        else:
            print(f"{i+1}. {name}  score: {score:.4f}")


if __name__ == '__main__':
    main()
