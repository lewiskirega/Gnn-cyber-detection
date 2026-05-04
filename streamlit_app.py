"""
Graphical interface for the GNN cyber-detection pipeline (Streamlit).

Run from the project root:
    streamlit run streamlit_app.py

Why this is used:
Provides an interactive front-end for experiments, suitable for demos and
thesis appendices (screenshots of configuration, curves, and result tables).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import ExperimentConfig, default_data_path, run_experiment
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from scripts.run_baselines import run_baselines


def main() -> None:
    st.set_page_config(
        page_title="GNN cyber detection",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("Robust GNN: coordinated attack detection")
    st.caption(
        "Flow mode: kNN graph over CIC-IDS flows. Host mode: IP graph when src_ip/dst_ip exist."
    )

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
        default_p = default_data_path()
        path_str = st.text_input(
            "Or path to CSV on disk",
            value=str(default_p) if default_p.is_file() else "",
            help="Used when no file is uploaded.",
        )

        st.header("Graph & model")
        graph_mode = st.selectbox("Graph mode", ["auto", "flows", "hosts"], index=0)
        label_col = st.text_input("Label column (flow datasets)", value="Label")
        max_samples = st.number_input("Max flow samples", min_value=1000, value=25_000, step=1000)
        knn_k = st.number_input("kNN k (flows)", min_value=3, value=15, step=1)
        epochs = st.number_input("Epochs", min_value=1, value=50, step=1)
        lr = st.number_input("Learning rate", min_value=1e-5, value=1e-2, format="%.5f")
        hidden = st.number_input("Hidden dim", min_value=8, value=16, step=8)
        dropout = st.slider("Dropout", 0.0, 0.9, 0.5)
        seed = st.number_input("Random seed", value=42, step=1)
        positive_class = st.text_input("Positive class name (optional)", value="")
        compare_baselines = st.checkbox("Run & Compare with ML Baselines", value=False)
        run_btn = st.button("Train & evaluate", type="primary")

    data_path: Path | None = None
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / f"gnn_cyber_{uploaded.name}"
        tmp.write_bytes(uploaded.getvalue())
        data_path = tmp
    elif path_str.strip():
        data_path = Path(path_str).expanduser()

    if not run_btn:
        st.info("Configure the sidebar and click **Train & evaluate**.")
        return

    if data_path is None or not data_path.is_file():
        st.error("Provide a valid CSV path or upload a file.")
        return

    pos = positive_class.strip() or None
    exp = ExperimentConfig(
        data_path=data_path,
        graph_mode=graph_mode,
        label_column=label_col if graph_mode != "hosts" else None,
        max_samples=int(max_samples),
        knn_k=int(knn_k),
        epochs=int(epochs),
        learning_rate=float(lr),
        hidden_channels=int(hidden),
        dropout=float(dropout),
        seed=int(seed),
        positive_class=pos,
    )

    progress = st.progress(0.0, text="Training…")
    status = st.empty()

    def on_epoch(epoch: int, loss: float, train_acc: float, val_acc: float) -> None:
        status.text(f"Epoch {epoch} — loss {loss:.4f} — val acc {val_acc:.4f}")
        progress.progress(min(1.0, epoch / max(1, exp.epochs)), text=f"Epoch {epoch}/{exp.epochs}")

    try:
        with st.spinner("Running experiment (this may take a while)…"):
            result = run_experiment(exp, epoch_callback=on_epoch)
            
            baseline_results = None
            if compare_baselines and exp.graph_mode in ["flows", "auto"]:
                with st.spinner("Running Baselines (Random Forest, Logistic Regression)..."):
                    lbl_col = exp.label_column or "Label"
                    baseline_results = run_baselines(data_path, lbl_col, exp.max_samples, exp.seed)
    except Exception as exc:  # pragma: no cover - UI feedback
        progress.empty()
        st.exception(exc)
        return

    progress.empty()
    status.empty()

    st.success(f"Test evaluated successfully ({result.graph_mode_used} graph on `{result.device}`)")
    st.write(result.info_message)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result.test_accuracy:.4f}")
    m2.metric("Precision", f"{result.test_precision:.4f}")
    m3.metric("Recall", f"{result.test_recall:.4f}")
    m4.metric("F1 Score", f"{result.test_f1:.4f}")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", result.num_nodes)
    c2.metric("Edges (index size)", result.num_edges)
    c3.metric("Input features", result.num_features)
    c4.metric("Classes", len(result.label_mapping))

    st.subheader("Label mapping")
    st.json(result.label_mapping)

    if result.training_history:
        st.subheader("Training curves")
        hist = pd.DataFrame(result.training_history)
        st.line_chart(hist.set_index("epoch")[["train_acc", "val_acc"]])
        st.line_chart(hist.set_index("epoch")[["loss"]])

    st.subheader("Sample test predictions")
    st.dataframe(result.sample_predictions, use_container_width=True)

    if 'baseline_results' in locals() and baseline_results:
        st.subheader("Baseline Comparison")
        all_results = baseline_results + [{
            "Model": "Proposed GNN", 
            "Accuracy": result.test_accuracy, 
            "Precision": result.test_precision, 
            "Recall": result.test_recall, 
            "F1-Score": result.test_f1
        }]
        comp_df = pd.DataFrame(all_results)
        st.dataframe(comp_df, use_container_width=True)
        st.bar_chart(comp_df.set_index("Model")[["F1-Score", "Accuracy"]])


if __name__ == "__main__":
    main()
