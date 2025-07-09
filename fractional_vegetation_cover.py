#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import joblib
import numpy as np
import optuna
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.features import geometry_mask
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Base directory for data
base_folder = os.getenv("BASE_FOLDER", r"C:\Users\Dell\Desktop\plant--vigor")

# Define survey names
surveys = {"survey1": "survey1", "survey2": "survey2"}

# Optical image paths for both surveys
fields = {
    "survey1": {
        "M2_Optical_Ortho": "/data/suparco/analysis/survey1/M2_Optical_Ortho.tif",
        "M3_Optical_Ortho": "/data/suparco/analysis/survey1/M3_Optical_Ortho.tif",
    },
    "survey2": {
        "M2_Optical_Ortho": "/data/suparco/analysis/survey2/M2_Optical_Ortho.tif",
        "M3_Optical_Ortho": "/data/suparco/analysis/survey2/M3_Optical_Ortho.tif",
    },
}

# Shapefile mappings for vegetation, soil, and impurities
shapefile_ortho_mapping = {
    "survey1": {
        "M2_Optical_Ortho": {
            "vegetation": os.path.join(base_folder, "vegetation", "survey1", "M2"),
            "soil": os.path.join(base_folder, "soil", "survey1", "M2"),
            "impurity": os.path.join(base_folder, "impurities", "survey1", "M2"),
        },
        "M3_Optical_Ortho": {
            "vegetation": os.path.join(base_folder, "vegetation", "survey1", "M3"),
            "soil": os.path.join(base_folder, "soil", "survey1", "M3"),
        },
    },
    "survey2": {
        "M2_Optical_Ortho": {
            "vegetation": os.path.join(base_folder, "vegetation", "survey2", "M2"),
            "soil": os.path.join(base_folder, "soil", "survey2", "M2"),
        },
        "M3_Optical_Ortho": {
            "vegetation": os.path.join(base_folder, "vegetation", "survey2", "M3"),
            "soil": os.path.join(base_folder, "soil", "survey2", "M3"),
        },
    },
}
def load_shapefiles(folder, label):
    if not os.path.exists(folder):
        return []
    shapefiles = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.shp')]
    return [(gpd.read_file(shp), label) for shp in shapefiles]

def prepare_training_data(shapefile, label, raster_path):
    try:
        with rasterio.open(raster_path) as src:
            raster = src.read()
            transform = src.transform
            height, width = raster.shape[1:]
            training_pixels, labels = [], []
            gdf = shapefile[0]

            for _, row in gdf.iterrows():
                mask = geometry_mask([row.geometry], transform=transform, 
                                   invert=True, out_shape=(height, width))
                pixels = raster[:, mask].T
                if pixels.size > 0:
                    training_pixels.append(pixels)
                    labels.append(np.full(len(pixels), label))

            if training_pixels:
                return np.vstack(training_pixels), np.hstack(labels)
            return np.array([]), np.array([])
    except Exception as e:
        print(f"Error processing {raster_path}: {str(e)}")
        return np.array([]), np.array([])

# Load all training data
X, y = [], []
for survey, field_data in shapefile_ortho_mapping.items():
    for field, shapefiles in field_data.items():
        raster_path = fields[survey].get(field)
        if raster_path and os.path.exists(raster_path):
            for key, folder in shapefiles.items():
                label = 1 if key == "vegetation" else 0 if key == "soil" else 2
                for shapefile in load_shapefiles(folder, label):
                    pixels, labels = prepare_training_data(shapefile, label, raster_path)
                    if pixels.size > 0:
                        X.append(pixels)
                        y.append(labels)

if not X or not y:
    raise ValueError("No training data found. Check shapefiles and raster paths.")

X, y = np.vstack(X), np.hstack(y)

# Split data - keep 20% for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))

def objective(trial, X_train, y_train):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    solver = trial.suggest_categorical("solver", ["saga", "lbfgs", "newton-cg"])
    C = trial.suggest_float("C", 1e-5, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 1000)
    l1_ratio = trial.suggest_float("l1_ratio", 0, 1) if penalty == "elasticnet" else None

    if solver in ["lbfgs", "newton-cg"] and penalty == "l1":
        raise optuna.TrialPruned()
    if solver != "saga" and penalty == "elasticnet":
        raise optuna.TrialPruned()

    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=C,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        multi_class="multinomial",
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, scoring='f1_weighted', n_jobs=-1
    )
    return np.mean(scores)

# Optimize study
study = optuna.create_study(direction="maximize")
study.optimize(
    lambda trial: objective(trial, X_train, y_train),
    n_trials=50,
    n_jobs=-1
)

# Train final model
best_params = study.best_params
final_model = LogisticRegression(
    **best_params,
    multi_class="multinomial",
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)

# Save model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final_model, os.path.join(model_dir, "logistic_regression_fvc_3.pkl"))

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Soil', 'Vegetation', 'Other']))
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Soil', 'Vegetation', 'Other'],
                yticklabels=['Soil', 'Vegetation', 'Other'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.show()
    plt.savefig(os.path.join(model_dir, "model_performance.png"))
    plt.close()

evaluate_model(final_model, X_test, y_test)

# Optuna visualizations
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
optuna.visualization.plot_contour(study, params=["C", "max_iter"]).show()


# In[3]:


def calculate_fvc_debug_with_graph(plot_shapefile, raster_path, clf, save_dir="FVC_Results"):
    """
    Calculates FVC using a trained classifier and plots the classification results.

    Parameters:
        plot_shapefile (str): Path to the plot shapefile.
        raster_path (str): Path to the raster image.
        clf (object): Trained classifier (e.g., RandomForest, SVM, etc.).
        save_dir (str): Directory to save results.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)

    # Read shapefile
    plot_gdf = gpd.read_file(plot_shapefile)

    # Read raster file
    with rasterio.open(raster_path) as src:
        raster = src.read()
        transform = src.transform
        height, width = src.height, src.width

        plot_ids, vegetation_counts, soil_counts = [], [], []
        fvc_percentages = []

        for _, row in plot_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=transform, invert=True, out_shape=(height, width))
            masked_pixels = raster[:, mask].T  # Convert to (n_samples, n_features)

            if masked_pixels.size > 0:
                predictions = clf.predict(masked_pixels)
                vegetation_pixel_count = np.sum(predictions == 1)
                soil_pixel_count = np.sum(predictions == 0)
            else:
                vegetation_pixel_count, soil_pixel_count = 0, 0

            plot_name = row.get("type", f"plot_{len(plot_ids)+1}")  # Fallback name
            plot_ids.append(plot_name)
            vegetation_counts.append(vegetation_pixel_count)
            soil_counts.append(soil_pixel_count)

            total_pixels = vegetation_pixel_count + soil_pixel_count
            fvc_percentage = (vegetation_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
            fvc_percentages.append(fvc_percentage)
            print(f"Plot: {plot_name} | FVC: {fvc_percentage:.2f}%")

        # Plot the bar graph
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Pixel count plot
        x = np.arange(len(plot_ids))
        ax1.bar(x - 0.2, soil_counts, 0.4, label='Soil Pixels', color='brown')
        ax1.bar(x + 0.2, vegetation_counts, 0.4, label='Vegetation Pixels', color='green')
        ax1.set_xlabel("Plot IDs")
        ax1.set_ylabel("Pixel Count")
        ax1.set_title("Pixel Classification for Each Plot")
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_ids, rotation=45, ha="right")
        ax1.legend()
        
        # FVC percentage plot
        ax2.bar(x, fvc_percentages, color='green', alpha=0.7)
        ax2.set_xlabel("Plot IDs")
        ax2.set_ylabel("FVC (%)")
        ax2.set_title("Fractional Vegetation Cover for Each Plot")
        ax2.set_xticks(x)
        ax2.set_xticklabels(plot_ids, rotation=45, ha="right")
        ax2.set_ylim(0, 100)  # FVC ranges from 0-100%
        
        # plt.tight_layout()
        # plt.show()

        # Save figure
        save_path = os.path.join(save_dir, "FVC_Classification.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Graph saved at: {save_path}")



# Load trained model (e.g.,Logistic regression)
clf = joblib.load(r"C:/Users/Dell/models/logistic_regression_fvc_3.pkl")

# Call function
calculate_fvc_debug_with_graph(r"C:/Users/Dell/Desktop/survey1/Optical/soil/survey1/M2/soil.shp", r"C:\Users\Dell\Desktop\survey1\Optical\ortho_data\M2_Optical_Ortho.tif", clf)

