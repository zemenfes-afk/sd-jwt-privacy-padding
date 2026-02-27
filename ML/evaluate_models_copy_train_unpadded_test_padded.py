import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import base64
import json
# --- The 5 Machine Learning Classifiers ---
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize


def generate_model_plots(models, X_train, X_test, y_train, y_test, jwttype, output_dir="model_plots"):
    """
    Train all models, evaluate, and save a comprehensive set of plots as PNG files.

    Plots generated:
        1. Accuracy comparison bar chart
        2. Confusion matrix per model
        3. Precision / Recall / F1 grouped bar chart
        4. ROC curves (one-vs-rest for multiclass)
        5. Precision-Recall curves
        6. Train vs Test accuracy (overfitting check)
        7. Combined summary dashboard

    Parameters
    ----------
    models   : dict  {name: sklearn_estimator}
    X_train, X_test, y_train, y_test : array-like splits
    output_dir : str  folder where PNGs are saved (created if absent)
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = np.unique(y_test)
    n_classes = len(classes)
    is_binary = (n_classes == 2)

    # ── Colours ────────────────────────────────────────────────────────────────
    PALETTE = sns.color_palette("husl", len(models))
    PLT_STYLE = "seaborn-v0_8-whitegrid"
    plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

    # ── Collect per-model metrics ───────────────────────────────────────────────
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        predictions = y_pred
        
        # Calculate Metrics
        accuracy = accuracy_score(y_test, predictions)
        correct_guesses = (predictions == y_test).sum()
        total_guesses = len(predictions)
        
        print(f"🔹 {name:<23} | Accuracy: {accuracy * 100:>6.2f}% | {correct_guesses:>3} / {total_guesses} correct")
    

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro = report["macro avg"]

        # Probability scores (if available)
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_test)
            if df.ndim == 1:
                y_score = np.column_stack([1 - df, df])
            else:
                y_score = df

        results[name] = {
            "y_pred":        y_pred,
            "y_score":       y_score,
            "accuracy":      accuracy_score(y_test, y_pred),
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "precision":     macro["precision"],
            "recall":        macro["recall"],
            "f1":            macro["f1-score"],
            "cm":            confusion_matrix(y_test, y_pred),
            "report":        report,
        }

    model_names = list(results.keys())
    colors = dict(zip(model_names, PALETTE))

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Accuracy Bar Chart
    # ══════════════════════════════════════════════════════════════════════════
    with plt.style.context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        accuracies = [results[n]["accuracy"] * 100 for n in model_names]
        bars = ax.barh(model_names, accuracies, color=[colors[n] for n in model_names],
                       edgecolor="white", linewidth=0.8, height=0.55)
        for bar, val in zip(bars, accuracies):
            ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.axvline(x=max(accuracies), linestyle="--", color="gray", alpha=0.5, linewidth=1)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"01_accuracy_comparison_{jwttype}.png"), bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Confusion Matrix — one file per model
    # ══════════════════════════════════════════════════════════════════════════
    for name in model_names:
        slug = name.lower().replace(" ", "_")
        with plt.style.context(PLT_STYLE):
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = results[name]["cm"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        cbar=False, linewidths=0.5, linecolor="white",
                        xticklabels=classes, yticklabels=classes)
            ax.set_title(f"{name}\nAcc: {results[name]['accuracy']*100:.2f}%",
                         fontsize=11, fontweight="bold")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"02_confusion_matrix_{slug}_{jwttype}.png"), bbox_inches="tight")
            plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Precision / Recall / F1 Grouped Bar Chart
    # ══════════════════════════════════════════════════════════════════════════
    with plt.style.context(PLT_STYLE):
        metrics = ["precision", "recall", "f1"]
        x = np.arange(len(model_names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        metric_colors = ["#4C72B0", "#55A868", "#C44E52"]

        for i, (metric, col) in enumerate(zip(metrics, metric_colors)):
            vals = [results[n][metric] for n in model_names]
            bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                          color=col, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score (Macro Avg)", fontsize=12)
        ax.set_title("Precision / Recall / F1 per Model", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"03_precision_recall_f1_{jwttype}.png"), bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. ROC Curves — one file per model + one combined
    # ══════════════════════════════════════════════════════════════════════════
    with plt.style.context(PLT_STYLE):
        fig_all, ax_all = plt.subplots(figsize=(8, 6))
        ax_all.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

        for name in model_names:
            slug = name.lower().replace(" ", "_")
            y_score = results[name]["y_score"]
            if y_score is None:
                continue
            try:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
                if is_binary:
                    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"AUC = {roc_auc:.3f}")
                    ax_all.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {roc_auc:.3f})")
                else:
                    y_bin = label_binarize(y_test, classes=classes)
                    fpr_all_c, tpr_all_c = [], []
                    for c in range(n_classes):
                        fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], y_score[:, c])
                        fpr_all_c.extend(fpr_c)
                        tpr_all_c.extend(tpr_c)
                    fpr_s = np.sort(np.unique(fpr_all_c))
                    tpr_mean = np.interp(fpr_s,
                                         np.array(fpr_all_c)[np.argsort(fpr_all_c)],
                                         np.array(tpr_all_c)[np.argsort(fpr_all_c)])
                    roc_auc = auc(fpr_s, tpr_mean)
                    ax.plot(fpr_s, tpr_mean, lw=2, color=colors[name], label=f"macro AUC ≈ {roc_auc:.3f}")
                    ax_all.plot(fpr_s, tpr_mean, lw=2, color=colors[name], label=f"{name} (AUC ≈ {roc_auc:.3f})")

                ax.set_xlabel("False Positive Rate", fontsize=12)
                ax.set_ylabel("True Positive Rate", fontsize=12)
                ax.set_title(f"ROC Curve — {name}", fontsize=13, fontweight="bold")
                ax.legend(loc="lower right", fontsize=10)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"04_roc_curve_{slug}_{jwttype}.png"), bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

        ax_all.set_xlabel("False Positive Rate", fontsize=12)
        ax_all.set_ylabel("True Positive Rate", fontsize=12)
        ax_all.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax_all.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        fig_all.savefig(os.path.join(output_dir, f"04_roc_curves_all_models_{jwttype}.png"), bbox_inches="tight")
        plt.close(fig_all)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Precision-Recall Curves — one file per model + one combined
    # ══════════════════════════════════════════════════════════════════════════
    with plt.style.context(PLT_STYLE):
        fig_all, ax_all = plt.subplots(figsize=(8, 6))

        for name in model_names:
            slug = name.lower().replace(" ", "_")
            y_score = results[name]["y_score"]
            if y_score is None:
                continue
            try:
                fig, ax = plt.subplots(figsize=(7, 5))
                if is_binary:
                    prec, rec, _ = precision_recall_curve(y_test, y_score[:, 1])
                    ap = average_precision_score(y_test, y_score[:, 1])
                    ax.plot(rec, prec, lw=2, color=colors[name], label=f"AP = {ap:.3f}")
                    ax_all.plot(rec, prec, lw=2, color=colors[name], label=f"{name} (AP = {ap:.3f})")
                else:
                    y_bin = label_binarize(y_test, classes=classes)
                    prec_list, rec_list = [], []
                    for c in range(n_classes):
                        p, r, _ = precision_recall_curve(y_bin[:, c], y_score[:, c])
                        prec_list.extend(p)
                        rec_list.extend(r)
                    sort_idx = np.argsort(rec_list)
                    ap = average_precision_score(y_bin, y_score, average="macro")
                    ax.plot(np.array(rec_list)[sort_idx], np.array(prec_list)[sort_idx],
                            lw=2, color=colors[name], label=f"macro AP ≈ {ap:.3f}")
                    ax_all.plot(np.array(rec_list)[sort_idx], np.array(prec_list)[sort_idx],
                                lw=2, color=colors[name], label=f"{name} (AP ≈ {ap:.3f})")

                ax.set_xlabel("Recall", fontsize=12)
                ax.set_ylabel("Precision", fontsize=12)
                ax.set_title(f"Precision-Recall Curve — {name}", fontsize=13, fontweight="bold")
                ax.legend(loc="upper right", fontsize=10)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"05_precision_recall_{slug}_{jwttype}.png"), bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

        ax_all.set_xlabel("Recall", fontsize=12)
        ax_all.set_ylabel("Precision", fontsize=12)
        ax_all.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
        ax_all.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        fig_all.savefig(os.path.join(output_dir, f"05_precision_recall_all_models_{jwttype}.png"), bbox_inches="tight")
        plt.close(fig_all)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Train vs Test Accuracy (Overfitting Check)
    # ══════════════════════════════════════════════════════════════════════════
    with plt.style.context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_names))
        width = 0.35

        train_accs = [results[n]["train_accuracy"] * 100 for n in model_names]
        test_accs  = [results[n]["accuracy"] * 100        for n in model_names]

        ax.bar(x - width / 2, train_accs, width, label="Train Accuracy",
               color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, test_accs,  width, label="Test Accuracy",
               color="#C44E52", alpha=0.85, edgecolor="white")

        for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
            gap = tr - te
            ax.text(i, max(tr, te) + 1, f"Δ{gap:.1f}%",
                    ha="center", fontsize=8, color="dimgray", style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Train vs Test Accuracy (Overfitting Check)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"06_train_vs_test_accuracy_{jwttype}.png"), bbox_inches="tight")
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Summary Dashboard
    # ══════════════════════════════════════════════════════════════════════════
    with plt.style.context(PLT_STYLE):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # --- Accuracy bar (top-left) ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.barh(model_names, [results[n]["accuracy"] * 100 for n in model_names],
                 color=[colors[n] for n in model_names], edgecolor="white", height=0.6)
        ax1.set_xlim(0, 110)
        ax1.set_xlabel("Accuracy (%)")
        ax1.set_title("Test Accuracy", fontweight="bold")

        # --- F1 bar (top-center) ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.barh(model_names, [results[n]["f1"] for n in model_names],
                 color=[colors[n] for n in model_names], edgecolor="white", height=0.6)
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel("F1 Score (Macro)")
        ax2.set_title("Macro F1 Score", fontweight="bold")

        # --- Train vs Test scatter (top-right) ---
        ax3 = fig.add_subplot(gs[0, 2])
        for name in model_names:
            ax3.scatter(results[name]["train_accuracy"] * 100,
                        results[name]["accuracy"] * 100,
                        color=colors[name], s=90, zorder=3, label=name)
            ax3.annotate(name.split()[0], (results[name]["train_accuracy"] * 100,
                                            results[name]["accuracy"] * 100),
                         textcoords="offset points", xytext=(4, 3), fontsize=7)
        lims = [50, 105]
        ax3.plot(lims, lims, "k--", lw=1, alpha=0.4, label="Perfect fit")
        ax3.set_xlim(*lims); ax3.set_ylim(*lims)
        ax3.set_xlabel("Train Accuracy (%)")
        ax3.set_ylabel("Test Accuracy (%)")
        ax3.set_title("Overfit Scatter", fontweight="bold")

        # --- Metrics heatmap (bottom row spans all 3 cols) ---
        ax4 = fig.add_subplot(gs[1, :])
        metric_keys = ["accuracy", "precision", "recall", "f1", "train_accuracy"]
        metric_labels = ["Test Acc", "Precision", "Recall", "F1", "Train Acc"]
        heat_data = np.array([[results[n][m] for m in metric_keys] for n in model_names])
        im = ax4.imshow(heat_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax4.set_xticks(range(len(metric_labels)))
        ax4.set_yticks(range(len(model_names)))
        ax4.set_xticklabels(metric_labels, fontsize=11)
        ax4.set_yticklabels(model_names, fontsize=10)
        for i in range(len(model_names)):
            for j in range(len(metric_keys)):
                ax4.text(j, i, f"{heat_data[i, j]:.3f}", ha="center", va="center",
                         fontsize=10, color="black", fontweight="bold")
        fig.colorbar(im, ax=ax4, fraction=0.02, pad=0.01)
        ax4.set_title("All Metrics Heatmap", fontweight="bold")

        fig.suptitle("🏆 Multi-Model Evaluation Dashboard", fontsize=16,
                     fontweight="bold", y=1.01)
        fig.savefig(os.path.join(output_dir, f"07_summary_dashboard_{jwttype}.png"), bbox_inches="tight")
        plt.close(fig)

    # ── Done ───────────────────────────────────────────────────────────────────
    saved = sorted(os.listdir(output_dir))
    print(f"\n✅ {len(saved)} plots saved to '{output_dir}/':")
    for f in saved:
        print(f"   📊 {f}")
    print()


# --- DIRECTORY CONFIGURATION ---
# Point this to your raw data
FHIR_DIR = "output/fhir/"
# Change this between your Unprotected and Protected folders to compare!
SDJWT_DIR = "output/final_sd_jwt_padded/" 
UNPADDED_SDJWT_DIR = "output/final_sd_jwt_unpadded/"

def find_disorder_count(data):
    data_arr=data.split('.')
    fin=''
    for datax in data_arr:
        try:
            text=datax
            while len(text)%4!=0:
                text=text+'='
            decoded = base64.b64decode(text).decode(encoding='utf-8', errors='ignore')
            if 'condition' in decoded:
                fin=decoded
                break
        except:
            pass
    try:
        findict=json.loads(fin)
        return len(findict['active_conditions'])
    except:
        pass

    return 0


def get_ground_truth(file_path):
    """
    Extracts the true number of active disorders from the raw FHIR JSON.
    This logic perfectly mirrors the generator script.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
            
        count = 0
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            if res.get("resourceType") == "Condition":
                coding = res.get("clinicalStatus", {}).get("coding", [])
                code = coding[0].get("code") if coding else None
                
                # Check if active (or no status provided)
                if (code and code.lower() == "active") or not code:
                    text = res.get("code", {}).get("text", "")
                    # CRITICAL: Must contain 'disorder' to match generator
                    if text and 'disorder' in text.lower():
                        count += 1
        return count
    except Exception as e:
        return 0

def evaluate_privacy_leakage(jwt_folder, jwttype):
    print(f"\n🕵️ Extracting metadata features from: {jwt_folder} ...")
    
    dataset = []
    fhir_files = [f for f in os.listdir(FHIR_DIR) if f.endswith(".json")]
    
    for filename in tqdm.tqdm(fhir_files):
        fhir_path = os.path.join(FHIR_DIR, filename)
        sdjwt_path = os.path.join(jwt_folder, filename.replace(".json", ".txt"))
        
        # Only process if the generated SD-JWT actually exists
        if os.path.exists(sdjwt_path):
            with open(sdjwt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                disorder_count=find_disorder_count(content)
                dataset.append({
                    "exact_byte_size": len(content),
                    "disorder_count": disorder_count,
                    "true_count": get_ground_truth(fhir_path)
                })

    if not dataset:
        print(f"❌ No tokens found in {jwt_folder}. Check your folder path.")
        return

    # --- DATA PREPARATION ---
    df = pd.DataFrame(dataset)
    
    # Define High-Risk threshold (> 5 active disorders)
    
    df['is_high_risk'] = (df['true_count'] > 5).astype(int)
    
    # ---------------------------------------------------------
    # ⚖️ NEW: UNDERSAMPLING FOR PERFECT 50/50 BALANCE
    # ---------------------------------------------------------
    # 1. Separate the two classes
    sick_patients = df[df['is_high_risk'] == 1]
    healthy_patients = df[df['is_high_risk'] == 0]
    
    # 2. Find out which group is smaller
    min_class_size = min(len(sick_patients), len(healthy_patients))
    
    # 3. Randomly sample the exact same amount from both groups
    sick_sampled = sick_patients.sample(n=min_class_size, random_state=42)
    healthy_sampled = healthy_patients.sample(n=min_class_size, random_state=42)
    
    # 4. Combine them into a perfectly balanced dataframe
    df_balanced = pd.concat([sick_sampled, healthy_sampled])
    
    print(f"⚖️ Dataset balanced: {min_class_size} High Risk vs {min_class_size} Low Risk")
    
    # Use the balanced dataset for the rest of the script
    X = df_balanced[['exact_byte_size', 'disorder_count']]
    y = df_balanced['is_high_risk']
    # ---------------------------------------------------------
    
    print(f"📊 Total balanced records processed: {len(df_balanced)}")

    return X, y

def run_models(X_train, X_test, y_train, y_test, jwttype):  
    
    # --- MODEL INITIALIZATION ---
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='linear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    print("\n" + "="*65)
    print("🏆 MULTI-MODEL ATTACK RESULTS")
    print("="*65)
    generate_model_plots(models, X_train, X_test, y_train, y_test, jwttype, output_dir="final_model_plots_diff_train")
    # --- TRAINING & EVALUATION LOOP ---
    #for name, model in models.items():
        # Train
    #    model.fit(X_train, y_train)
        
        # Predict
    #    predictions = model.predict(X_test)
        
        # Calculate Metrics
     #   accuracy = accuracy_score(y_test, predictions)
      #  correct_guesses = (predictions == y_test).sum()
       # total_guesses = len(predictions)
        
        #print(f"🔹 {name:<23} | Accuracy: {accuracy * 100:>6.2f}% | {correct_guesses:>3} / {total_guesses} correct")
        
    print("="*65 + "\n")

if __name__ == "__main__":
    print("Running for UNPADDED")
    unpadded_x, unpadded_y=evaluate_privacy_leakage(UNPADDED_SDJWT_DIR, "unpadded")
    print("Running for PADDED")
    padded_x, padded_y=evaluate_privacy_leakage(SDJWT_DIR, "padded")

    print("Running Models")
    run_models(unpadded_x, padded_x, unpadded_y, padded_y, "jwts_comparison")

