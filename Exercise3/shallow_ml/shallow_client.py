import sys
import os
import cv2
import numpy as np
import time
import random
import pandas as pd
import argparse
from typing import Tuple, List, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
from data.load_cifar import retrieve_all_cifar

IMG_SIZE = (64, 64)
VOCAB_SIZE = 100
SEED = 42

# Default Paths (Can be overridden by flags if needed, or hardcoded here)
DEFAULT_GTSRB_TRAIN = r"C:\Users\lenar\Downloads\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images"
DEFAULT_GTSRB_TEST = r"C:\Users\lenar\Downloads\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images"

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_gtsrb_data(base_path: str, mode: str = "train", sample_fraction: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(base_path):
        print(f"[ERROR] Path not found: {base_path}")
        return np.array([]), np.array([])

    images, labels = [], []
    
    if mode == "train":
        classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        for class_id, class_name in enumerate(classes):
            class_dir = os.path.join(base_path, class_name)
            img_files = [f for f in os.listdir(class_dir) if f.endswith(('.ppm', '.png'))]
            
            if sample_fraction < 1.0:
                count = int(len(img_files) * sample_fraction)
                img_files = random.sample(img_files, max(1, count))

            for img_file in img_files:
                img = cv2.imread(os.path.join(class_dir, img_file))
                if img is not None:
                    images.append(cv2.resize(img, IMG_SIZE))
                    labels.append(class_id)

    elif mode == "test":
        csv_file = None
        search_paths = [base_path, os.path.dirname(base_path)]
        for p in search_paths:
            for f in os.listdir(p):
                if f.endswith(".csv") and "test" in f.lower():
                    csv_file = os.path.join(p, f)
                    break
            if csv_file: break
        
        if csv_file is None:
            print("[ERROR] No GT-final_test.csv found for test data.")
            return np.array([]), np.array([])
        
        try:
            df = pd.read_csv(csv_file, sep=';')
            if 'ClassId' not in df.columns:
                print("[ERROR] CSV missing ClassId.")
                return np.array([]), np.array([])
        except:
            return np.array([]), np.array([])

        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=SEED)

        for index, row in df.iterrows():
            img_path = os.path.join(base_path, row['Filename'])
            img = cv2.imread(img_path)
            if img is not None:
                images.append(cv2.resize(img, IMG_SIZE))
                labels.append(row['ClassId'])

    return np.array(images), np.array(labels)

def load_cifar_adapter(sample_fraction: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    raw_data = retrieve_all_cifar("train") 
    X_flat = np.array(raw_data['data'])
    y = np.array(raw_data['labels'])
    
    if sample_fraction < 1.0:
        indices = np.random.choice(len(y), int(len(y)*sample_fraction), replace=False)
        X_flat = X_flat[indices]
        y = y[indices]

    X_images = X_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_processed = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), IMG_SIZE) for img in X_images]
    return np.array(X_processed), y

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def extract_color_histograms(images: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    features = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        features.append(hist.flatten())
    return np.array(features)

class VisualBagOfWords:
    def __init__(self, k=100):
        self.k = k
        self.kmeans = None
        self.sift = cv2.SIFT_create()

    def fit(self, images: np.ndarray):
        print("   [BoW] Building vocabulary...")
        descriptor_list = []
        sample_indices = np.random.choice(len(images), min(len(images), 1000), replace=False)
        for idx in sample_indices:
            gray = cv2.cvtColor(images[idx], cv2.COLOR_BGR2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            if des is not None: descriptor_list.append(des)
        
        if descriptor_list:
            all_descriptors = np.vstack(descriptor_list)
            self.kmeans = KMeans(n_clusters=self.k, random_state=SEED, n_init=10).fit(all_descriptors)

    def transform(self, images: np.ndarray) -> np.ndarray:
        histograms = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            hist = np.zeros(self.k)
            if des is not None and self.kmeans:
                preds = self.kmeans.predict(des)
                for p in preds: hist[p] += 1
            if np.sum(hist) > 0: hist /= np.sum(hist)
            histograms.append(hist)
        return np.array(histograms)

# ==========================================
# 3. MAIN LOGIC (ARGPARSE)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Image Classification Benchmarking Tool")

    # --- Configuration Flags ---
    parser.add_argument("--dataset", type=str, required=True, choices=['GTSRB', 'CIFAR'], help="Dataset to use")
    parser.add_argument("--sample_frac", type=float, default=0.1, help="Fraction of data to use (0.0 - 1.0)")
    parser.add_argument("--models", nargs='+', default=['ALL'], 
                        choices=['RF_HIST', 'RF_SIFT', 'SVM_HIST', 'SVM_SIFT', 'ALL'],
                        help="List of models to run (e.g. --models RF_HIST SVM_SIFT)")
    
    # --- Hyperparameters (RF) ---
    parser.add_argument("--n_estimators", type=int, default=100, help="RF: Number of trees")
    parser.add_argument("--max_depth", type=int, default=None, help="RF: Max depth of trees")
    
    # --- Hyperparameters (SVM) ---
    parser.add_argument("--C", type=float, default=1.0, help="SVM: Regularization parameter")
    parser.add_argument("--gamma", type=str, default='scale', help="SVM: Kernel coefficient (scale, auto, or float)")

    # --- Result Output ---
    parser.add_argument("--result", type=str, default='all', 
                        choices=['accuracy', 'precision', 'recall', 'f1', 'all'],
                        help="Metric to print out")

    args = parser.parse_args()

    # Handle 'ALL' model selection
    if 'ALL' in args.models:
        selected_models = ['RF_HIST', 'RF_SIFT', 'SVM_HIST', 'SVM_SIFT']
    else:
        selected_models = args.models

    # Convert gamma to float if possible (svm handles 'scale'/'auto' strings)
    try:
        svm_gamma = float(args.gamma)
    except ValueError:
        svm_gamma = args.gamma

    print(f"\n{'='*50}")
    print(f"CONFIGURATION")
    print(f"Dataset:      {args.dataset}")
    print(f"Sample Frac:  {args.sample_frac}")
    print(f"Models:       {selected_models}")
    print(f"Result Type:  {args.result}")
    print(f"RF Params:    n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"SVM Params:   C={args.C}, gamma={svm_gamma}")
    print(f"{'='*50}\n")

    # 1. LOAD DATA
    print("[INFO] Loading Data...")
    if args.dataset == 'GTSRB':
        X_train, y_train = load_gtsrb_data(DEFAULT_GTSRB_TRAIN, mode="train", sample_fraction=args.sample_frac)
        X_test, y_test = load_gtsrb_data(DEFAULT_GTSRB_TEST, mode="test", sample_fraction=args.sample_frac)
    else:
        X_all, y_all = load_cifar_adapter(sample_fraction=args.sample_frac)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

    if len(X_train) == 0:
        print("[ERROR] No data loaded. Exiting.")
        return

    # 2. FEATURE EXTRACTION (Only compute what is needed)
    needs_hist = any("HIST" in m for m in selected_models)
    needs_sift = any("SIFT" in m for m in selected_models)

    X_train_hist, X_test_hist = None, None
    X_train_sift, X_test_sift = None, None

    if needs_hist:
        print("[INFO] Extracting Color Histograms...")
        start = time.time()
        X_train_hist = extract_color_histograms(X_train)
        X_test_hist = extract_color_histograms(X_test)
        print(f"       Done in {time.time()-start:.2f}s")

    if needs_sift:
        print("[INFO] Extracting SIFT BoW...")
        start = time.time()
        bow = VisualBagOfWords(k=VOCAB_SIZE)
        bow.fit(X_train)
        X_train_sift = bow.transform(X_train)
        X_test_sift = bow.transform(X_test)
        print(f"       Done in {time.time()-start:.2f}s")

    # 3. TRAINING & EVALUATION LOOP
    print(f"\n{'='*50}")
    print("MODEL EVALUATION")

    for model_name in selected_models:
        print(f"\n Running {model_name}")
        
        # A. Setup Model & Data
        if "RF" in model_name:
            clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=SEED, n_jobs=-1)
        else: # SVM
            clf = SVC(C=args.C, gamma=svm_gamma, probability=False, random_state=SEED) # prob=False is faster

        if "HIST" in model_name:
            X_tr, X_te = X_train_hist, X_test_hist
        else: # SIFT
            X_tr, X_te = X_train_sift, X_test_sift

        # B. Train
        start_train = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - start_train

        # C. Predict
        y_pred = clf.predict(X_te)

        # D. Output Results
        print(f"   Training Time: {train_time:.2f}s")
        
        if args.result == 'all':
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=3))
            print("-" * 30)
        else:
            score = 0.0
            if args.result == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif args.result == 'precision':
                score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            elif args.result == 'recall':
                score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            elif args.result == 'f1':
                score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"   {args.result.upper()}: {score:.4f}")

if __name__ == "__main__":
    main()