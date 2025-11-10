import sys
from PIL import Image
import os
from tqdm import tqdm
import pickle
from img2vec import Img2Vec


def img2vec_converter(model, cuda, fine_tuned_path=None, input_dir='resized'):
    """Convert images found in input_dir to vectors using Img2Vec.
    Returns dictionary mapping filename -> vector.
    """
    img2vec = Img2Vec(cuda=cuda, model=model, fine_tuned_path=fine_tuned_path)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found")

    vectors = {}
    files = sorted(os.listdir(input_dir))
    print(f"Converting {len(files)} images from '{input_dir}' to vectors using model={model}")
    for image in tqdm(files):
        img_path = os.path.join(input_dir, image)
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue
        try:
            vec = img2vec.get_vec(img, normalize_vec=True)
        except RuntimeError:
            vec = img2vec.get_vec(img.convert('RGB'), normalize_vec=True)
        assert len(vec) == img2vec.layer_output_size
        vectors[image] = vec
        img.close()

    print("Saving image vectors")
    os.makedirs(os.path.join("pickles", model), exist_ok=True)
    file = open(os.path.join('pickles', model, 'vectors.pkl'), 'wb')
    pickle.dump(vectors, file)
    file.close()

    return vectors