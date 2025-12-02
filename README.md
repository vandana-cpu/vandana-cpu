# Content-Based Image Retrieval (CBIR)

Simple Streamlit UI for content-based image retrieval using a pretrained ResNet50 feature extractor.

- UI: [app.py](app.py) — main application.
- Dependencies: [requirements.txt](requirements.txt)
- Training / dataset folder: [Train/](Train)

## Quickstart

The project works best inside a Python virtual environment. Example commands used by the author:

1. Create a virtual environment and activate it:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   Note: `faiss-cpu` and `annoy` are optional; remove them from [requirements.txt](requirements.txt) if they fail to install on your platform.

3. Run the app:
   ```sh
   streamlit run app.py
   ```

Open the Streamlit UI in the provided URL printed by Streamlit.

## Project layout

- app.py — Streamlit app that:
  - builds the model via [`build_resnet50`](app.py),
  - extracts per-image descriptors using [`extract_single_feature`](app.py) and [`extract_features_from_paths`](app.py),
  - lists images with [`list_image_files`](app.py),
  - searches with [`knn_euclid`](app.py), [`knn_cosine`](app.py), [`knn_faiss_index`](app.py) (if installed), and [`knn_annoy_index`](app.py).
- requirements.txt — Python package dependencies.
- Train/ — expected dataset folder; organize images into subfolders (example folders already present: `Wonders_of_World/chichen_itza`, `Wonders_of_World/pyramids_of_giza`, `Wonders_of_World/roman_colosseum`).

## How to use the app

1. Tab "1. Extract / Train features"
   - Provide the path to your image folder (default `./Train`).
   - Click "Extract features and save" to compute descriptors and save a compressed `.npz` containing `paths` + `features`.

2. Tab "2. Query similar images"
   - Upload a query image or provide a local path.
   - Choose similarity method: `euclid`, `cosine`, or (if available) `faiss` / `annoy`.
   - Click "Search" to find top-K similar images from the cached features file.

## Notes & tips

- Feature extractor: ResNet50 with `include_top=False` and `pooling='avg'` (produces 2048-d vectors).
- The features cache format is a compressed NumPy `.npz` with `paths` and `features`.
- If you have a GPU and want TensorFlow GPU builds, replace the CPU `tensorflow` package accordingly.
- If `faiss-cpu` fails to install, remove it from [requirements.txt](requirements.txt) and rely on `euclid`/`cosine`/`annoy`.

## Files & Symbols

- [app.py](app.py)
- [requirements.txt](requirements.txt)
- [Train/](Train)

Referenced symbols in the code:
- [`build_resnet50`](app.py)
- [`extract_single_feature`](app.py)
- [`extract_features_from_paths`](app.py)
- [`list_image_files`](app.py)
- [`knn_euclid`](app.py)
- [`knn_cosine`](app.py)
- [`knn_faiss_index`](app.py)
- [`knn_annoy_index`](app.py)

## License

Add your preferred license file (e.g. `LICENSE`) or mention project-specific license terms here.
