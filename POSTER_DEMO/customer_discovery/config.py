from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PipelineConfig:
    random_state: int = 42

    # Cleaning / feature engineering
    drop_columns: List[str] = field(default_factory=list)
    categorical_cols: Optional[List[str]] = None
    numeric_cols: Optional[List[str]] = None

    # Segmentation
    k_min: int = 2
    k_max: int = 10
    kmeans_n_init: int = 20

    # Prediction
    label_col: str = "converted"
    test_size: float = 0.2
    model_type: str = "random_forest"  # or "logistic_regression"
