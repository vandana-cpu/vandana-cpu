"""
Streamlit UI for CBIR (Content-Based Image Retrieval)

Features:
- Extract features from a folder of images and save to a .npz cache
- Query for similar images by uploading a query image or providing a query image path
- Choose method: euclid | cosine | faiss (if installed) | annoy (if installed)
- Show results in a grid with distances / similarity scores

Run:
    streamlit run app.py
"""

import os
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
import tempfile
import time

# UI will need the model & extraction utilities. We'll embed minimal implementations here
# to keep this file self-contained. If you have separate modules (feature_extractor.py, search.py, utils.py)
# you can import them instead.

# ---- Optional: try to import faiss / annoy ----
try:
    import faiss
    _has_faiss = True
except Exception:
    faiss = None
    _has_faiss = False

try:
    from annoy import AnnoyIndex
    _has_annoy = True
except Exception:
    AnnoyIndex = None
    _has_annoy = False

# ---- TensorFlow Keras ResNet50 extractor ----
@st.cache_resource(show_spinner=False)
def build_resnet50():
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    # Using include_top=False and pooling='avg' gives fixed-length descriptor
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

def extract_single_feature(model, img_pil: Image.Image, target_size=(224,224)):
    """
    Given a PIL.Image, resize and extract ResNet50 features.
    Returns 1D numpy array.
    """
    from tensorflow.keras.preprocessing import image as kimage
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import numpy as np

    img = img_pil.convert("RGB")
    img = img.resize(target_size)
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = model.predict(x, verbose=0)
    feats = np.asarray(feats).reshape(-1)
    return feats

def extract_features_from_paths(image_paths: List[str], model, progress_callback=None) -> np.ndarray:
    import numpy as np
    feats = []
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            f = extract_single_feature(model, img)
            feats.append(f)
        except Exception as e:
            # skip unreadable images
            print("skip", p, e)
            continue
        if progress_callback:
            progress_callback(i+1, n)
    if len(feats) == 0:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.vstack(feats)

# ---- Simple similarity functions ----
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def knn_euclid(query_vec, features, k=5):
    dists = euclidean_distances([query_vec], features)[0]
    idx = np.argsort(dists)[:k]
    return idx, dists[idx]

def knn_cosine(query_vec, features, k=5):
    sims = cosine_similarity([query_vec], features)[0]
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def knn_faiss_index(features, query_vecs, k=5, metric='L2'):
    if not _has_faiss:
        raise RuntimeError("faiss not installed.")
    features = features.astype('float32')
    if metric == 'L2':
        index = faiss.IndexFlatL2(features.shape[1])
    else:
        index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    D, I = index.search(query_vecs.astype('float32'), k)
    return I, D

def knn_annoy_index(features, query_vecs, k=5, metric='euclidean', n_trees=10):
    if not _has_annoy:
        raise RuntimeError("annoy not installed.")
    f = features.shape[1]
    metric_name = 'euclidean' if metric == 'euclidean' else 'angular'
    t = AnnoyIndex(f, metric_name)
    for i in range(len(features)):
        t.add_item(i, features[i].astype('float32').tolist())
    t.build(n_trees)
    results = []
    distances = []
    for q in query_vecs:
        res = t.get_nns_by_vector(q.astype('float32').tolist(), k, include_distances=True)
        if isinstance(res, tuple):
            inds, dists = res
        else:
            inds = res
            dists = []
        results.append(inds)
        distances.append(dists)
    return np.array(results), np.array(distances)

# ---- Utilities ----
def list_image_files(folder: str, exts=(".jpg",".jpeg",".png",".bmp")):
    p = Path(folder)
    if not p.exists():
        return []
    files = [str(x) for x in sorted(p.rglob("*")) if x.suffix.lower() in exts]
    return files

def save_features_npz(path: str, image_paths: List[str], features: np.ndarray):
    np.savez_compressed(path, paths=np.array(image_paths), features=features.astype(np.float32))

def load_features_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return list(d["paths"].astype(str)), np.array(d["features"])

# ---- Streamlit UI layout ----
st.set_page_config(page_title="Content Based Image Retrieval", layout="wide")
st.title("Content-Based Image Retrieval — UI")

# Sidebar controls
st.sidebar.header("Data & Options")
data_dir = st.sidebar.text_input("Image folder (Train) path", value="./Train")
features_file = st.sidebar.text_input("Features cache path", value="./features.npz")
top_k = st.sidebar.number_input("Top K results", min_value=1, max_value=50, value=5)
method = st.sidebar.selectbox("Similarity method", options=["euclid", "cosine", "faiss" if _has_faiss else "faiss (not installed)", "annoy" if _has_annoy else "annoy (not installed)"])
use_faiss = method == "faiss" and _has_faiss
use_annoy = method == "annoy" and _has_annoy

st.sidebar.markdown("---")
st.sidebar.write("Model: ResNet50 pre-trained (ImageNet), pooling='avg'")

# Main tabs: Extract / Query
tab1, tab2 = st.tabs(["1. Extract / Train features", "2. Query similar images"])

# ---- TAB 1: Extract ----
with tab1:
    st.header("Extract features from a folder")
    st.write("This will compute image descriptors for every image in the folder and save to a `.npz` file (paths + features).")
    col1, col2 = st.columns([3,1])
    with col1:
        st.text("Detected images in folder:")
        detected_list = list_image_files(data_dir)
        st.write(f"Found {len(detected_list)} images")
        # show up to first 6 images
        cols = st.columns(6)
        for i, p in enumerate(detected_list[:6]):
            with cols[i]:
                try:
                    img = Image.open(p).convert("RGB")
                    st.image(img, use_column_width=True, caption=os.path.basename(p))
                except Exception:
                    st.write("img error")
    with col2:
        st.write("Extraction options")
        override_cache = st.checkbox("Overwrite existing features file", value=False)
        extract_btn = st.button("Extract features and save")

    if extract_btn:
        if len(detected_list) == 0:
            st.warning("No images found in the provided folder. Please check the path.")
        else:
            # Confirm writing
            if os.path.exists(features_file) and not override_cache:
                st.warning(f"Features file already exists at {features_file}. Check 'Overwrite existing features file' to recompute.")
            else:
                st.info("Building model (if not cached) and extracting features...")
                model = build_resnet50()
                progress = st.progress(0)
                n = len(detected_list)
                # small progress callback
                def progress_cb(done, total):
                    progress.progress(int(done/total*100))
                features = extract_features_from_paths(detected_list, model, progress_callback=progress_cb)
                save_features_npz(features_file, detected_list, features)
                st.success(f"Saved {features.shape[0]} features to {features_file}")
                st.balloons()

# ---- TAB 2: Query ----
with tab2:
    st.header("Query similar images")
    st.write("Upload a query image (or provide a local path). Then click 'Search' to find similar images from the cached features.")
    colq1, colq2 = st.columns([2,1])
    with colq1:
        uploaded_file = st.file_uploader("Upload query image (recommended)", type=["jpg","jpeg","png", "webp"])
        query_path_input = st.text_input("Or provide path to query image", value="")
    with colq2:
        show_results = st.checkbox("Show result images", value=True)
        search_btn = st.button("Search")

    if search_btn:
        # load features
        if not os.path.exists(features_file):
            st.error(f"Features file not found: {features_file}. Extract first (Tab 1).")
        else:
            image_paths, features = load_features_npz(features_file)
            st.info(f"Loaded {len(image_paths)} indexed images.")
            # Build query image PIL
            if uploaded_file is not None:
                try:
                    img_bytes = uploaded_file.read()
                    query_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    # Save a temp copy for path-display (optional)
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    tmpf.write(img_bytes)
                    tmpf.flush()
                    query_display_path = tmpf.name
                except Exception as e:
                    st.error(f"Could not read uploaded image: {e}")
                    query_img = None
                    query_display_path = None
            elif query_path_input.strip():
                if os.path.exists(query_path_input):
                    query_img = Image.open(query_path_input).convert("RGB")
                    query_display_path = query_path_input
                else:
                    st.error("Provided query path does not exist.")
                    query_img = None
                    query_display_path = None
            else:
                st.error("No query image provided.")
                query_img = None
                query_display_path = None

            if query_img is not None:
                st.image(query_img, caption="Query image", width=250)
                # extract feature
                with st.spinner("Extracting query feature..."):
                    model = build_resnet50()
                    qf = extract_single_feature(model, query_img)
                # search
                method_actual = method.split()[0]  # account for "(not installed)" label
                st.write(f"Using method: {method_actual}")
                t0 = time.time()
                try:
                    if method_actual == "euclid":
                        idx, d = knn_euclid(qf, features, k=top_k)
                        results = [(image_paths[i], float(dv)) for i, dv in zip(idx, d)]
                    elif method_actual == "cosine":
                        idx, s = knn_cosine(qf, features, k=top_k)
                        results = [(image_paths[i], float(sv)) for i, sv in zip(idx, s)]
                    elif method_actual == "faiss":
                        if not _has_faiss:
                            st.error("Faiss is not installed.")
                            results = []
                        else:
                            I, D = knn_faiss_index(features, np.array([qf]), k=top_k, metric='L2')
                            idx = I[0]
                            results = [(image_paths[i], float(D[0][j])) for j,i in enumerate(idx)]
                    elif method_actual == "annoy":
                        if not _has_annoy:
                            st.error("Annoy is not installed.")
                            results = []
                        else:
                            I, D = knn_annoy_index(features, np.array([qf]), k=top_k, metric='euclidean', n_trees=10)
                            idx = I[0]
                            # Annoy returns distances maybe empty depending on version
                            results = []
                            for j, i in enumerate(idx):
                                try:
                                    dist = float(D[0][j]) if D.size > 0 else None
                                except Exception:
                                    dist = None
                                results.append((image_paths[i], dist))
                    else:
                        st.error("Unknown method selected.")
                        results = []
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    results = []
                t1 = time.time()
                st.write(f"Search completed in {t1-t0:.3f}s — found {len(results)} results")

                if len(results) > 0:
                    # Display results table and images
                    st.subheader("Results")
                    rows = []
                    for path, score in results:
                        rows.append({"path": path, "score": score})
                    st.table(rows)

                    if show_results:
                        cols = st.columns(min(5, len(results)))
                        for i, (path, score) in enumerate(results):
                            col = cols[i % len(cols)]
                            try:
                                img = Image.open(path).convert("RGB")
                                caption = f"{os.path.basename(path)}\n{score:.4f}" if score is not None else os.path.basename(path)
                                col.image(img, use_column_width=True, caption=caption)
                            except Exception as e:
                                col.write(f"Err opening {path}")
