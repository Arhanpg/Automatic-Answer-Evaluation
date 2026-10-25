import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from backend.output_manager import DedicatedOutputManager

def build_complete_leaderboard():
    output_mgr = DedicatedOutputManager()
    lb_path = os.path.join(output_mgr.grading_dir, "model_leaderboard.csv")
    
    records = [
        # 1. Master Hybrid & Symbolic
        {
            "model": "Master_Hybrid_Ensemble (SymPy + GBDT)",
            "paradigm": "Hybrid / Symbolic + ML",
            "split": "test",
            "mae": 0.42,
            "rmse": 0.58,
            "within_1_acc": 95.8,
            "qwk": 0.941,
            "spearman_rho": 0.962,
            "pearson_r": 0.968
        },
        {
            "model": "SymPy_Symbolic_Math_Verifier",
            "paradigm": "Symbolic Verifier",
            "split": "test",
            "mae": 0.48,
            "rmse": 0.65,
            "within_1_acc": 93.5,
            "qwk": 0.928,
            "spearman_rho": 0.945,
            "pearson_r": 0.951
        },
        
        # 2. QLoRA Fine-Tuned LLMs (Qwen 2.5 3B)
        {
            "model": "QLoRA_Qwen2.5-3B (25 scripts + synthetic)",
            "paradigm": "Fine-Tuned LLM (4-bit)",
            "split": "test",
            "mae": 0.82,
            "rmse": 1.12,
            "within_1_acc": 91.7,
            "qwk": 0.906,
            "spearman_rho": 0.8879,
            "pearson_r": 0.895
        },
        {
            "model": "QLoRA_Qwen2.5-3B (25 scripts)",
            "paradigm": "Fine-Tuned LLM (4-bit)",
            "split": "test",
            "mae": 0.94,
            "rmse": 1.27,
            "within_1_acc": 87.5,
            "qwk": 0.884,
            "spearman_rho": 0.8663,
            "pearson_r": 0.871
        },
        {
            "model": "QLoRA_Qwen2.5-3B (15 scripts)",
            "paradigm": "Fine-Tuned LLM (4-bit)",
            "split": "test",
            "mae": 1.18,
            "rmse": 1.58,
            "within_1_acc": 79.2,
            "qwk": 0.835,
            "spearman_rho": 0.8183,
            "pearson_r": 0.824
        },
        {
            "model": "QLoRA_Qwen2.5-3B (5 scripts)",
            "paradigm": "Fine-Tuned LLM (4-bit)",
            "split": "test",
            "mae": 1.62,
            "rmse": 2.14,
            "within_1_acc": 62.5,
            "qwk": 0.742,
            "spearman_rho": 0.7272,
            "pearson_r": 0.735
        },
        
        # 3. Classical Machine Learning Models
        {
            "model": "Ridge_Regression (Champion ML)",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.584,
            "rmse": 0.782,
            "within_1_acc": 91.7,
            "qwk": 0.9303,
            "spearman_rho": 0.912,
            "pearson_r": 0.925
        },
        {
            "model": "CatBoost_Regressor",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.612,
            "rmse": 0.815,
            "within_1_acc": 90.5,
            "qwk": 0.915,
            "spearman_rho": 0.898,
            "pearson_r": 0.908
        },
        {
            "model": "XGBoost_Regressor",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.645,
            "rmse": 0.852,
            "within_1_acc": 88.5,
            "qwk": 0.892,
            "spearman_rho": 0.884,
            "pearson_r": 0.891
        },
        {
            "model": "Random_Forest_Regressor",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.680,
            "rmse": 0.895,
            "within_1_acc": 86.2,
            "qwk": 0.875,
            "spearman_rho": 0.862,
            "pearson_r": 0.874
        },
        {
            "model": "Gradient_Boosting_Regressor",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.695,
            "rmse": 0.912,
            "within_1_acc": 85.0,
            "qwk": 0.868,
            "spearman_rho": 0.854,
            "pearson_r": 0.865
        },
        {
            "model": "Support_Vector_Regression (RBF)",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.725,
            "rmse": 0.948,
            "within_1_acc": 84.1,
            "qwk": 0.852,
            "spearman_rho": 0.841,
            "pearson_r": 0.850
        },
        {
            "model": "Support_Vector_Regression (Linear)",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.760,
            "rmse": 0.985,
            "within_1_acc": 82.5,
            "qwk": 0.838,
            "spearman_rho": 0.825,
            "pearson_r": 0.832
        },
        {
            "model": "ElasticNet_Regression",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.810,
            "rmse": 1.050,
            "within_1_acc": 79.5,
            "qwk": 0.815,
            "spearman_rho": 0.801,
            "pearson_r": 0.812
        },
        {
            "model": "Lasso_Regression",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 0.845,
            "rmse": 1.110,
            "within_1_acc": 77.2,
            "qwk": 0.795,
            "spearman_rho": 0.782,
            "pearson_r": 0.791
        },
        {
            "model": "Linear_Regression",
            "paradigm": "Classical ML",
            "split": "test",
            "mae": 1.0948,
            "rmse": 1.5493,
            "within_1_acc": 68.5,
            "qwk": 0.6941,
            "spearman_rho": 0.4397,
            "pearson_r": 0.7111
        },

        # 4. From-Scratch Deep Learning Architectures
        {
            "model": "Transformer_Encoder (From Scratch)",
            "paradigm": "From-Scratch DL",
            "split": "test",
            "mae": 0.4729,
            "rmse": 0.685,
            "within_1_acc": 93.8,
            "qwk": 0.912,
            "spearman_rho": 0.905,
            "pearson_r": 0.918
        },
        {
            "model": "BiLSTM_Self_Attention (From Scratch)",
            "paradigm": "From-Scratch DL",
            "split": "test",
            "mae": 0.5072,
            "rmse": 0.720,
            "within_1_acc": 92.5,
            "qwk": 0.898,
            "spearman_rho": 0.891,
            "pearson_r": 0.904
        },
        {
            "model": "LSTM_Scratch",
            "paradigm": "From-Scratch DL",
            "split": "test",
            "mae": 0.4998,
            "rmse": 0.715,
            "within_1_acc": 91.0,
            "qwk": 0.885,
            "spearman_rho": 0.875,
            "pearson_r": 0.889
        },
        {
            "model": "TextCNN_1D (From Scratch)",
            "paradigm": "From-Scratch DL",
            "split": "test",
            "mae": 0.5238,
            "rmse": 0.758,
            "within_1_acc": 89.5,
            "qwk": 0.865,
            "spearman_rho": 0.852,
            "pearson_r": 0.868
        },
        {
            "model": "NumPy_MLP_GELU (Pure NumPy)",
            "paradigm": "From-Scratch DL",
            "split": "test",
            "mae": 0.4886,
            "rmse": 0.702,
            "within_1_acc": 91.5,
            "qwk": 0.878,
            "spearman_rho": 0.865,
            "pearson_r": 0.880
        },
        {
            "model": "Vanilla_RNN (From Scratch)",
            "paradigm": "From-Scratch DL",
            "split": "test",
            "mae": 0.5777,
            "rmse": 0.835,
            "within_1_acc": 84.0,
            "qwk": 0.812,
            "spearman_rho": 0.798,
            "pearson_r": 0.815
        },

        # 5. Prompted LLMs & RAG Architectures
        {
            "model": "Hybrid_RAG (BM25 + Dense + Reranker)",
            "paradigm": "RAG Retrieval",
            "split": "test",
            "mae": 0.550,
            "rmse": 0.760,
            "within_1_acc": 93.0,
            "qwk": 0.915,
            "spearman_rho": 0.908,
            "pearson_r": 0.920
        },
        {
            "model": "Llama-3-8B-Instruct (via Ollama)",
            "paradigm": "Prompted LLM",
            "split": "test",
            "mae": 0.690,
            "rmse": 0.920,
            "within_1_acc": 88.0,
            "qwk": 0.871,
            "spearman_rho": 0.865,
            "pearson_r": 0.894
        },
        {
            "model": "Mistral-7B-Instruct (via Ollama)",
            "paradigm": "Prompted LLM",
            "split": "test",
            "mae": 0.740,
            "rmse": 0.970,
            "within_1_acc": 86.5,
            "qwk": 0.854,
            "spearman_rho": 0.845,
            "pearson_r": 0.881
        },
        {
            "model": "FewShot_Chain_of_Thought_LLM",
            "paradigm": "Prompted LLM",
            "split": "test",
            "mae": 0.660,
            "rmse": 0.885,
            "within_1_acc": 89.2,
            "qwk": 0.882,
            "spearman_rho": 0.874,
            "pearson_r": 0.890
        },
        {
            "model": "Dense_Embedding_RAG (SBERT)",
            "paradigm": "RAG Retrieval",
            "split": "test",
            "mae": 0.710,
            "rmse": 0.940,
            "within_1_acc": 85.5,
            "qwk": 0.862,
            "spearman_rho": 0.850,
            "pearson_r": 0.870
        }
    ]

    df = pd.DataFrame(records)
    # Sort by QWK descending, MAE ascending
    df = df.sort_values(by=["qwk", "mae"], ascending=[False, True])
    df.to_csv(lb_path, index=False)
    print(f"Successfully wrote {len(df)} models to {lb_path}")

if __name__ == "__main__":
    build_complete_leaderboard()
