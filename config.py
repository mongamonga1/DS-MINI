"""main.py와 노트북이 함께 사용하는 실험 설정."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ESS_DATA_DIR", PROJECT_ROOT / "data" / "raw")
        ).expanduser()
    )
    batch_files: tuple[str, ...] = (
        "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    )
    download_urls: dict[str, str] = field(default_factory=dict)
    random_state: int = 42
    valid_size: int = 10
    cv_folds: int = 5
    target_transforms: tuple[str, ...] = ("original", "log10")
    model_names: tuple[str, ...] = ("elastic_net", "ridge", "random_forest")
    selection_metric: str = "cv_mape_mean"
    top_k_stage1: int = 3
    top_k_stage2: int = 3
    run_final_evaluation: bool = True
    random_forest_trees: int = 100
    default_model_params: dict[str, dict[str, float | int]] = field(
        default_factory=lambda: {
            "elastic_net": {"alpha": 0.1, "l1_ratio": 0.5},
            "ridge": {"alpha": 1.0},
            "random_forest": {"max_depth": 3, "min_samples_leaf": 3},
        }
    )
    tuning_grids: dict[str, dict[str, tuple[float | int, ...]]] = field(
        default_factory=lambda: {
            "elastic_net": {
                "alpha": (0.001, 0.01, 0.1, 1.0),
                "l1_ratio": (0.2, 0.5, 0.8),
            },
            "ridge": {"alpha": (0.1, 1.0, 10.0, 100.0)},
            "random_forest": {
                "max_depth": (2, 3, 5),
                "min_samples_leaf": (2, 3, 5),
            },
        }
    )
    corrupted_temperature_cells: tuple[str, ...] = (
        "2018-02-20_batchdata_updated_struct_errorcorrect:cell038",
    )
    target_mape_pct: float = 9.1

    def validate(self) -> None:
        if self.random_state < 0:
            raise ValueError("random_state는 0 이상이어야 합니다.")
        if self.valid_size <= 0 or self.cv_folds < 2:
            raise ValueError("검증 셀 수와 CV 수를 확인하세요.")
        if self.top_k_stage1 < 1 or self.top_k_stage2 < 1:
            raise ValueError("단계별 생존 후보 수는 1 이상이어야 합니다.")
        if set(self.model_names) != set(self.default_model_params):
            raise ValueError("모델 후보와 기본 매개변수의 모델 이름이 다릅니다.")
        if set(self.model_names) != set(self.tuning_grids):
            raise ValueError("모델 후보와 탐색 범위의 모델 이름이 다릅니다.")


CONFIG = ExperimentConfig()
