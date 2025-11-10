# Vehicle Recognition System
# VehicleRecognitionSystem

A small toolkit for image (vehicle) embedding, similarity search and visualization. 

This repository contains utilities to:

- Resize and prepare an image dataset (`resize_images.py`, `download.py`).
- Convert images into embeddings using CNN backbones (`vectorize.py`, `img2vec.py`).
- Build and save a cosine similarity matrix and top-k nearest lists (`similarity_matrix.py`, `cos_sim.py`, `top_similar.py`).
- Query a single image for similar images (`similarity_search.py`).
- Produce a visualization of similar images (`plot_similar.py`, `plotfunctions.py`).
- Compute a t-SNE map of embeddings for visualization (`tsne_map.py`).
- Train / fine-tune models for classification or contrastive embeddings (`finetune_arcface.py`, `finetune_contrasive_512_addition.py`).

The project stores intermediate artifacts under the `pickles/` folder and resized images under `resized/`.

## Quick overview

- Input images: put original images in `images/` (or provide a dataset to `download.py`).
- Resized images: `resized/` (224×224) created by `resize_images.py` or `download.py`.
- Embeddings: `pickles/<model>/vectors.pkl` produced by `vectorize.py`.
- Similarity matrix: `pickles/<model>/sim_matrix.pkl` produced by `similarity_matrix.py`.
- Top-k similar lists: `pickles/<model>/similar_names.pkl` and `similar_values.pkl`.

## Requirements

The project uses Python and the following packages (see `requirements.txt`):

- torch, torchvision
- timm
- numpy, pandas
- scikit-learn
- pillow
- matplotlib
- tqdm

Install requirements (recommended to use a virtual environment):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install timm matplotlib tqdm
```

Note: `requirements.txt` lists base packages; `timm` and `matplotlib` may need to be installed separately depending on your environment.

## Typical workflow (quickstart)

1. Prepare images

	- If you have a dataset organized as `Dataset/<Make>/<Model>/*.jpg`, you can use `download.py` to collect and resize images and create metadata `pickles/jsondata.pkl`:

	```powershell
	python .\download.py -d D:\path\to\archive -n 5000
	```

	- Or if you already have images in `images/`, resize them:

	```powershell
	python .\resize_images.py
	```

2. Vectorize images

	- Create L2-normalized embeddings for all images using a chosen backbone (default `resnet50d` or `resnet18` in some scripts):

	```powershell
	python .\vectorize.py
	# This script exposes the core function `img2vec_converter(model, cuda, fine_tuned_path, input_dir)`
	```

	The script writes `pickles/<model>/vectors.pkl` (a dict mapping filename -> vector).

3. Build similarity matrix and top-k lists

	```powershell
	python .\similarity_matrix.py -m resnet50d --cuda False -i resized
	```

	This will compute the cosine similarity matrix and save:

	- `pickles/<model>/sim_matrix.pkl`
	- `pickles/<model>/similar_names.pkl`
	- `pickles/<model>/similar_values.pkl`

4. Query a single image (similarity_search)

	```powershell
	python .\similarity_search.py -i resized\000001_img.jpg -m resnet50d -k 10
	```

	- Use `--fine-tuned path/to/checkpoint.pth` to load a checkpoint when extracting the query embedding.

5. Visualize top similar images

	```powershell
	python .\plot_similar.py -i 000001_img.jpg -m resnet50d
	```

	Output will be saved in `similar_images/<query_image>` as a PNG showing the query and its top matches.

6. t-SNE map

	```powershell
	python .\tsne_map.py --model resnet50d --n-samples 2000 --perplexity 30 --out pickles\resnet50d\tsne_resnet50d.png
	```

	This will produce `pickles/<model>/tsne_embeddings.pkl` and a PNG scatter plot.

## Training / Fine-tuning

- `finetune_arcface.py` — example training script using an ArcFace margin layer. It trains a classification-style head and saves `best_model.pth` when validation accuracy improves.
- `finetune_contrasive_512_addition.py` — contrastive training pipeline producing a contrastive embedding model (default projection to 512-D). It includes training/validation loops, threshold selection, and saves a checkpoint (configurable via `--output`).

These are example scripts: adapt paths, batch sizes and dataset locations to your environment. Training scripts expect dataset layout like `Dataset/<Make>/<Model>/*.jpg`.

Security note: loading arbitrary PyTorch checkpoint files (via `torch.load`) can execute pickled code. `img2vec.py` attempts a safer `weights_only` load if available, and the fine-tune scripts save reasonably structured checkpoints — still only load checkpoints from trusted sources.

## Files and important functions

- `img2vec.py` — Img2Vec class wrapping timm/torchvision backbones and exposing `get_vec(img, normalize_vec=True)`.
- `vectorize.py` — convenience wrapper that iterates `resized/` images and writes `pickles/<model>/vectors.pkl`.
- `cos_sim.py` — `matrix_calculator(vectors)` returns a pandas DataFrame cosine similarity matrix.
- `similarity_matrix.py` — CLI to vectorize and compute/save similarity matrix and top-K lists.
- `similarity_search.py` — CLI to query a single image and print top-K results.
- `getsimilar.py` / `top_similar.py` — helpers to load top-N similar names/values.
- `plot_similar.py`, `plotfunctions.py` — plotting utilities that use `pickles/jsondata.pkl` (if present) to show metadata labels.
- `tsne_map.py` — compute and visualize t-SNE embeddings from saved vectors.

## Output layout (examples)

- `resized/` — resized images (224×224)
- `pickles/<model>/vectors.pkl` — serialized dict filename -> vector
- `pickles/<model>/sim_matrix.pkl` — pandas DataFrame cosine similarity
- `pickles/<model>/similar_names.pkl` — DataFrame, per-row top names
- `pickles/<model>/similar_values.pkl` — DataFrame, per-row top similarity scores
- `pickles/jsondata.pkl` — optional metadata used by plotting utilities
- `similar_images/` — generated plots for query images

## Troubleshooting & notes

- If `vectorize.py` fails due to GPU memory or CUDA availability, pass `--cuda False` or set the `cuda` argument to False when calling the functions.
- When a fine-tuned model checkpoint does not directly match a backbone, `img2vec.py` includes logic to strip common prefixes like `module.` or `backbone.` and attempt a partial load with `strict=False`.
- Use smaller `--n-samples` for `tsne_map.py` to speed up t-SNE.

## Contribution

Feel free to open issues or pull requests. Useful improvements:

- Add a small CLI entrypoint script wiring these steps into a single pipeline.
- Add unit tests for core utilities (`img2vec`, `cos_sim`, etc.).
- Provide Docker/Conda environment for easier reproducibility.

## License

Include your preferred license file (e.g., `LICENSE`). Currently no license file is included in the repository.

---

If you'd like, I can:

- Commit this README in the repo (already prepared),
- Add example PowerShell scripts to run the common workflows, or
- Generate a small `scripts/` folder with simple wrappers for Linux/Windows.

Tell me which you'd like next.