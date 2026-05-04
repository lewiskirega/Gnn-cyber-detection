"""
Data loading module for the GNN cyber-attack detection system.

This module is responsible for:
1) Reading network traffic records from CSV files.
2) Validating the schema expected by downstream graph/model components.
3) Preparing labels and feature columns in a consistent format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Configuration object that centralizes dataset column names.

    Why this is used:
    Keeping schema definitions in one place avoids hard-coded strings across
    the codebase and makes experiments easier when datasets use different names.
    """

    src_column: str = "src_ip"
    dst_column: str = "dst_ip"
    label_column: str = "label"
    features_column: str = "features"


class TrafficDataLoader:
    """
    Loads and validates traffic datasets for graph-based cyber detection.

    Why this is used:
    A dedicated loader keeps data concerns isolated from model/training code,
    which improves maintainability, reproducibility, and academic clarity.
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None) -> None:
        self.config = config or DataLoaderConfig()

    def read_csv_table(self, file_path: str | Path) -> pd.DataFrame:
        """
        Read a CSV and normalize column names (strip whitespace).

        Why this is used:
        Public datasets such as CIC-IDS2017 often use headers like ' Label' with
        leading spaces; normalizing avoids silent mismatches in column names.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = pd.read_csv(path)
        data.columns = data.columns.str.strip()
        return data

    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        """
        Read a CSV file into a DataFrame and validate required columns.

        Why this is used:
        All downstream stages (graph builder and trainer) assume a minimum
        schema; failing fast here prevents hidden runtime issues later.
        """

        data = self.read_csv_table(file_path)
        self._validate_required_columns(data)
        return data

    def infer_feature_columns(
        self,
        data: pd.DataFrame,
        explicit_feature_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Determine which columns should be used as numeric edge features.

        Why this is used:
        Feature inference allows flexible datasets: users can either provide a
        fixed feature list or rely on automatic detection for rapid prototyping.
        """

        if explicit_feature_columns:
            missing = [col for col in explicit_feature_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Explicit feature columns not found in dataset: {missing}")
            return explicit_feature_columns

        reserved = {
            self.config.src_column,
            self.config.dst_column,
            self.config.label_column,
        }

        # We keep only numeric columns because they can be consumed directly
        # by ML models without additional encoding.
        inferred = [
            col
            for col in data.select_dtypes(include=["number"]).columns
            if col not in reserved
        ]
        return inferred

    def encode_labels(
        self,
        data: pd.DataFrame,
        positive_class_name: str = "attack",
    ) -> Tuple[pd.Series, Dict[str, int]]:
        """
        Convert labels to integer class IDs for classification training.

        Why this is used:
        Neural network loss functions (e.g., CrossEntropyLoss) expect integer
        targets; this method standardizes label encoding for that requirement.
        """

        label_col = self.config.label_column
        unique_labels = sorted(data[label_col].dropna().astype(str).unique().tolist())
        if not unique_labels:
            raise ValueError("Label column is empty after removing missing values.")

        # For binary settings, we force a stable mapping where 'attack' is 1
        # (if present) to keep experiment interpretation consistent.
        if len(unique_labels) == 2 and positive_class_name in unique_labels:
            negative_label = [lbl for lbl in unique_labels if lbl != positive_class_name][0]
            label_mapping = {negative_label: 0, positive_class_name: 1}
        else:
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        encoded = data[label_col].astype(str).map(label_mapping)
        if encoded.isna().any():
            raise ValueError("Some labels could not be encoded. Check label values.")

        return encoded.astype(int), label_mapping

    def prepare_dataset(
        self,
        file_path: str | Path,
        explicit_feature_columns: Optional[List[str]] = None,
        positive_class_name: str = "attack",
    ) -> Tuple[pd.DataFrame, List[str], pd.Series, Dict[str, int]]:
        """
        End-to-end data preparation routine used by training/inference pipelines.

        Returns:
            data: Raw validated DataFrame.
            feature_columns: Selected numeric feature columns.
            encoded_labels: Integer labels aligned with `data` rows.
            label_mapping: Dictionary from original label string to class ID.

        Why this is used:
        A single entry point ensures all experiments apply identical loading and
        preprocessing logic, which supports repeatable research results.
        """

        data = self.load_csv(file_path)
        feature_columns = self.infer_feature_columns(data, explicit_feature_columns)
        encoded_labels, label_mapping = self.encode_labels(
            data=data,
            positive_class_name=positive_class_name,
        )
        return data, feature_columns, encoded_labels, label_mapping

    def infer_flow_feature_columns(
        self,
        data: pd.DataFrame,
        explicit_feature_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Select numeric columns to use as node features for flow-level graphs.

        Why this is used:
        CIC-IDS flow CSVs have many statistical columns but no IP fields; each
        row becomes one graph node, so we need a clean numeric feature list.
        """

        if explicit_feature_columns:
            missing = [c for c in explicit_feature_columns if c not in data.columns]
            if missing:
                raise ValueError(f"Explicit feature columns not found in dataset: {missing}")
            return explicit_feature_columns

        label_col = self.config.label_column
        inferred = [
            col
            for col in data.select_dtypes(include=["number"]).columns
            if col != label_col
        ]
        if not inferred:
            raise ValueError(
                "No numeric feature columns found for flow graph. "
                "Check the CSV or pass explicit_feature_columns."
            )
        return inferred

    @staticmethod
    def _stratified_sample_dataframe(
        dataframe: pd.DataFrame,
        label_column: str,
        max_rows: int,
        random_state: int,
    ) -> pd.DataFrame:
        """
        Downsample a labeled table while keeping all classes represented.

        Why this is used:
        kNN graph construction scales with sample count; stratified subsampling
        keeps class balance for DDoS vs BENIGN experiments.
        """

        if len(dataframe) <= max_rows:
            return dataframe

        classes = dataframe[label_column].dropna().unique()
        parts: List[pd.DataFrame] = []
        per_class = max(1, max_rows // len(classes))
        for cls in classes:
            subset = dataframe[dataframe[label_column] == cls]
            take = min(per_class, len(subset))
            parts.append(subset.sample(n=take, random_state=random_state))
        out = pd.concat(parts, ignore_index=True)
        if len(out) > max_rows:
            out = out.sample(n=max_rows, random_state=random_state)
        return out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    def prepare_flow_dataset(
        self,
        file_path: str | Path,
        explicit_feature_columns: Optional[List[str]] = None,
        positive_class_name: str = "DDoS",
        max_samples: Optional[int] = 25_000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
        """
        Load a flow-record CSV (row = one flow) and return feature matrix + labels.

        Returns:
            X: 2D float array of shape (n_flows, n_features), cleaned and finite.
            y: 1D int labels aligned with rows of X.
            feature_columns: Names of columns used in X (for reporting).
            label_mapping: Original string label -> class id.

        Why this is used:
        Some exports (including many CIC-IDS2017 MachineLearningCSV drops) omit
        IP columns; a similarity graph over flows still lets a GCN aggregate
        local neighborhoods in feature space for attack detection.
        """

        data = self.read_csv_table(file_path)
        label_col = self.config.label_column
        if label_col not in data.columns:
            raise ValueError(
                f"Flow dataset must include label column {label_col!r}. "
                f"Found columns: {list(data.columns)[:12]} ..."
            )

        feature_columns = self.infer_flow_feature_columns(data, explicit_feature_columns)
        for col in feature_columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=feature_columns + [label_col])
        if max_samples is not None and len(data) > max_samples:
            data = self._stratified_sample_dataframe(
                data, label_col, max_samples, random_state
            )

        encoded_labels, label_mapping = self.encode_labels(
            data=data,
            positive_class_name=positive_class_name,
        )
        x_matrix = data[feature_columns].to_numpy(dtype=np.float64)
        x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        y_vector = encoded_labels.to_numpy(dtype=np.int64)
        return x_matrix, y_vector, feature_columns, label_mapping

    def _validate_required_columns(self, data: pd.DataFrame) -> None:
        """
        Validate that the required schema is present in the input DataFrame.

        Why this is used:
        Coordinated attack detection needs source, destination, and label fields;
        missing any of them makes graph construction or supervision invalid.
        """

        required = {
            self.config.src_column,
            self.config.dst_column,
            self.config.label_column,
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")
