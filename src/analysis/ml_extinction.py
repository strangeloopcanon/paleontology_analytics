import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def run_ml_extinction_analysis(data_path="data/processed/merged_occurrences.parquet", output_dir="data/analysis"):
    """
    Machine Learning analysis to predict extinction risk.
    
    Features per genus per time bin:
    - geographic_range: number of unique localities
    - abundance: occurrence count
    - lat_range: latitudinal extent
    - env_breadth: number of unique environments
    - age: how long has this genus existed (number of previous bins present)
    
    Target: extinct_next_bin (1 if genus disappears in next time bin)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data for ML extinction analysis from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Filter valid data
    df = df.dropna(subset=["mid_ma", "genus", "lat", "lng"])
    
    # Create time bins (5 Ma)
    df["time_bin"] = (df["mid_ma"] / 5).round() * 5
    
    # Create locality identifier
    df["locality"] = list(zip((df["lat"] / 5).round() * 5, (df["lng"] / 5).round() * 5))
    
    # Get sorted unique time bins (oldest to youngest)
    time_bins = sorted(df["time_bin"].unique(), reverse=True)  # High Ma = older
    
    # Build feature matrix
    records = []
    
    print("Engineering features per genus per time bin...")
    for i, current_bin in enumerate(time_bins[:-1]):  # Skip the last (youngest) bin - no "next" bin
        next_bin = time_bins[i + 1]
        
        current_data = df[df["time_bin"] == current_bin]
        next_data = df[df["time_bin"] == next_bin]
        
        current_genera = set(current_data["genus"].unique())
        next_genera = set(next_data["genus"].unique())
        
        for genus in current_genera:
            genus_data = current_data[current_data["genus"] == genus]
            
            # Features
            geographic_range = genus_data["locality"].nunique()
            abundance = len(genus_data)
            lat_range = genus_data["lat"].max() - genus_data["lat"].min()
            
            # Environment breadth (if available)
            if "environment" in genus_data.columns:
                env_breadth = genus_data["environment"].nunique()
            else:
                env_breadth = 0
            
            # Age: count how many OLDER bins this genus appears in
            older_bins = [b for b in time_bins if b > current_bin]
            age = sum(1 for b in older_bins if genus in df[df["time_bin"] == b]["genus"].values)
            
            # Target: did this genus go extinct (not found in next bin)?
            extinct_next_bin = 1 if genus not in next_genera else 0
            
            records.append({
                "genus": genus,
                "time_bin": current_bin,
                "geographic_range": geographic_range,
                "abundance": abundance,
                "lat_range": lat_range,
                "env_breadth": env_breadth,
                "age": age,
                "extinct_next_bin": extinct_next_bin
            })
    
    features_df = pd.DataFrame(records)
    
    if len(features_df) < 100:
        print(f"Insufficient data for ML analysis ({len(features_df)} samples). Need at least 100.")
        return
    
    print(f"Built feature matrix with {len(features_df)} genus-bin samples.")
    print(f"Extinction rate: {features_df['extinct_next_bin'].mean():.2%}")
    
    # Prepare ML data
    feature_cols = ["geographic_range", "abundance", "lat_range", "env_breadth", "age"]
    X = features_df[feature_cols].fillna(0)
    y = features_df["extinct_next_bin"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.5  # If only one class in test set
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Survived', 'Extinct'])}")
    
    # Feature Importances
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=True)
    
    # Plot Feature Importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"], color='crimson')
    plt.xlabel("Feature Importance")
    plt.title("What Predicts Extinction? (Random Forest Feature Importances)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "extinction_feature_importances.png"))
    print(f"\nFeature importances plot saved.")
    
    # Save summary
    summary_path = os.path.join(output_dir, "ml_extinction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== ML Extinction Prediction Summary ===\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Total genus-bin samples: {len(features_df)}\n")
        f.write(f"Extinction rate (per bin): {features_df['extinct_next_bin'].mean():.2%}\n\n")
        f.write(f"Model: Random Forest (n_estimators=100, balanced)\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"ROC-AUC: {roc_auc:.3f}\n\n")
        f.write("Feature Importances (descending):\n")
        for _, row in importance_df.sort_values("importance", ascending=False).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    run_ml_extinction_analysis()
