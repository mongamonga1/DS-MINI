"""공통 전처리, 교차검증, 후보 선별, 최종 학습·평가."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline

from config import ExperimentConfig
from .data import SplitBundle
from .features import FeatureSet


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    stage: str
    feature_set: str
    family: str
    variant: str
    model_name: str
    target_transform: str
    params: tuple[tuple[str, float | int], ...]

    @property
    def param_dict(self) -> dict[str, float | int]:
        return dict(self.params)


@dataclass
class CandidateEvaluation:
    summary: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame


class CommonPreprocessor(BaseEstimator, TransformerMixin):
    """학습 fold에서 공통 열의 결측 대체·표준화 기준을 계산합니다."""

    def __init__(self, columns: tuple[str, ...]):
        self.columns = columns

    def fit(self, features: pd.DataFrame, target=None):
        frame = features.loc[:, list(self.columns)].replace([np.inf, -np.inf], np.nan)
        self.active_columns_ = frame.columns[frame.notna().any(axis=0)].tolist()
        if not self.active_columns_:
            raise ValueError("학습 fold에서 모든 공통 피처가 결측입니다.")
        active = frame[self.active_columns_]
        self.medians_ = active.median(axis=0)
        filled = active.fillna(self.medians_)
        self.means_ = filled.mean(axis=0)
        self.scales_ = filled.std(axis=0, ddof=0).replace(0, 1.0)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = features.loc[:, self.active_columns_].replace([np.inf, -np.inf], np.nan)
        filled = frame.fillna(self.medians_)
        return (filled - self.means_) / self.scales_


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: tuple[str, ...]):
        self.columns = columns

    def fit(self, features: pd.DataFrame, target=None):
        missing = sorted(set(self.columns) - set(features.columns))
        if missing:
            raise ValueError(f"학습 fold에서 전체 결측인 선택 피처: {missing}")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.loc[:, list(self.columns)]


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape or actual_array.size == 0:
        raise ValueError("실제값과 예측값의 모양을 확인하세요.")
    if not np.isfinite(actual_array).all() or not np.all(actual_array > 0):
        raise ValueError("수명 라벨은 유한한 양수여야 합니다.")
    if not np.isfinite(predicted_array).all():
        raise ValueError("유한하지 않은 예측값이 있습니다.")
    errors = predicted_array - actual_array
    return {
        "mape_pct": float(np.mean(np.abs(errors) / actual_array) * 100),
        "mae_cycles": float(np.mean(np.abs(errors))),
        "rmse_cycles": float(np.sqrt(np.mean(np.square(errors)))),
    }


def inverse_log10(values):
    return np.power(10.0, values)


def _parameter_text(params: dict[str, float | int]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_candidate(
    stage: str,
    feature_set: FeatureSet,
    model_name: str,
    target_transform: str,
    params: dict[str, float | int],
) -> Candidate:
    parameter_text = _parameter_text(params)
    candidate_id = "|".join(
        [stage, feature_set.name, model_name, target_transform, parameter_text]
    )
    return Candidate(
        candidate_id=candidate_id,
        stage=stage,
        feature_set=feature_set.name,
        family=feature_set.family,
        variant=feature_set.variant,
        model_name=model_name,
        target_transform=target_transform,
        params=tuple(sorted(params.items())),
    )


def build_initial_candidates(
    feature_sets: dict[str, FeatureSet], config: ExperimentConfig
) -> list[Candidate]:
    return [
        make_candidate(
            "stage1",
            feature_set,
            model_name,
            target_transform,
            config.default_model_params[model_name],
        )
        for feature_set in feature_sets.values()
        for model_name in config.model_names
        for target_transform in config.target_transforms
    ]


def parameter_combinations(
    grid: dict[str, tuple[float | int, ...]]
) -> list[dict[str, float | int]]:
    names = tuple(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*(grid[name] for name in names))
    ]


def build_tuned_candidates(
    survivors: list[Candidate],
    feature_sets: dict[str, FeatureSet],
    config: ExperimentConfig,
) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}
    for survivor in survivors:
        feature_set = feature_sets[survivor.feature_set]
        for params in parameter_combinations(config.tuning_grids[survivor.model_name]):
            candidate = make_candidate(
                "stage3_tuning",
                feature_set,
                survivor.model_name,
                survivor.target_transform,
                params,
            )
            candidates[candidate.candidate_id] = candidate
    return list(candidates.values())


def _build_regressor(candidate: Candidate, config: ExperimentConfig):
    params = candidate.param_dict
    if candidate.model_name == "elastic_net":
        return ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=100000,
            tol=1e-4,
            random_state=config.random_state,
        )
    if candidate.model_name == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if candidate.model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=config.random_forest_trees,
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=config.random_state,
            n_jobs=1,
        )
    raise ValueError(f"알 수 없는 모델입니다: {candidate.model_name}")


def build_pipeline(
    candidate: Candidate,
    feature_set: FeatureSet,
    all_feature_columns: tuple[str, ...],
    config: ExperimentConfig,
):
    pipeline = Pipeline(
        [
            ("preprocess", CommonPreprocessor(all_feature_columns)),
            ("select", FeatureSelector(feature_set.columns)),
            ("model", _build_regressor(candidate, config)),
        ]
    )
    if candidate.target_transform == "log10":
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log10,
            inverse_func=inverse_log10,
            check_inverse=False,
        )
    return pipeline


def _fold_partitions(
    train_table: pd.DataFrame, split: SplitBundle
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    fold_map = split.cv_folds.set_index("cell_id")["validation_fold"]
    fold_numbers = train_table["cell_id"].map(fold_map)
    if fold_numbers.isna().any():
        raise ValueError("학습 셀에 CV fold가 지정되지 않았습니다.")
    return [
        (
            fold_number,
            np.flatnonzero(fold_numbers.to_numpy() != fold_number),
            np.flatnonzero(fold_numbers.to_numpy() == fold_number),
        )
        for fold_number in sorted(fold_numbers.unique())
    ]


def evaluate_candidates(
    candidates: list[Candidate],
    feature_sets: dict[str, FeatureSet],
    feature_table: pd.DataFrame,
    split: SplitBundle,
    all_feature_columns: tuple[str, ...],
    config: ExperimentConfig,
    progress=None,
) -> CandidateEvaluation:
    train_table = (
        feature_table.loc[feature_table["cell_id"].isin(split.train_ids)]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    if len(train_table) != 36:
        raise ValueError("학습용 셀 수가 36개가 아닙니다.")
    partitions = _fold_partitions(train_table, split)
    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for candidate_number, candidate in enumerate(candidates, start=1):
        candidate_fold_rows: list[dict] = []
        candidate_predictions: list[dict] = []
        status = "ok"
        error_message = ""
        convergence_warnings = 0
        try:
            feature_set = feature_sets[candidate.feature_set]
            for fold_number, fit_indices, validation_indices in partitions:
                fit_table = train_table.iloc[fit_indices]
                validation_table = train_table.iloc[validation_indices]
                estimator = build_pipeline(
                    candidate, feature_set, all_feature_columns, config
                )
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(
                        fit_table.loc[:, list(all_feature_columns)],
                        fit_table["cycle_life"],
                    )
                convergence_warnings += sum(
                    issubclass(item.category, ConvergenceWarning) for item in captured
                )
                predicted = estimator.predict(
                    validation_table.loc[:, list(all_feature_columns)]
                )
                metrics = regression_metrics(validation_table["cycle_life"], predicted)
                candidate_fold_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "stage": candidate.stage,
                        "fold": fold_number,
                        "n_fit": len(fit_table),
                        "n_validation": len(validation_table),
                        "nonpositive_predictions": int(np.sum(predicted <= 0)),
                        **metrics,
                    }
                )
                for cell_id, actual, prediction in zip(
                    validation_table["cell_id"],
                    validation_table["cycle_life"],
                    predicted,
                    strict=True,
                ):
                    candidate_predictions.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "stage": candidate.stage,
                            "fold": fold_number,
                            "cell_id": cell_id,
                            "actual": float(actual),
                            "predicted": float(prediction),
                        }
                    )
        except Exception as error:
            status = "failed"
            error_message = f"{type(error).__name__}: {error}"

        if status == "ok" and convergence_warnings:
            status = "warning"
            error_message = f"수렴 경고 {convergence_warnings}회"
        if status == "ok":
            fold_frame = pd.DataFrame(candidate_fold_rows)
            summary_metrics = {
                "cv_mape_mean": fold_frame["mape_pct"].mean(),
                "cv_mape_std": fold_frame["mape_pct"].std(ddof=0),
                "cv_mae_mean": fold_frame["mae_cycles"].mean(),
                "cv_rmse_mean": fold_frame["rmse_cycles"].mean(),
            }
            fold_rows.extend(candidate_fold_rows)
            prediction_rows.extend(candidate_predictions)
        else:
            summary_metrics = {
                "cv_mape_mean": np.nan,
                "cv_mape_std": np.nan,
                "cv_mae_mean": np.nan,
                "cv_rmse_mean": np.nan,
            }
        feature_set = feature_sets[candidate.feature_set]
        summary_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "stage": candidate.stage,
                "feature_set": candidate.feature_set,
                "family": candidate.family,
                "variant": candidate.variant,
                "model": candidate.model_name,
                "target_transform": candidate.target_transform,
                "params": _parameter_text(candidate.param_dict),
                "n_features": len(feature_set.columns),
                "status": status,
                "reason": error_message,
                **summary_metrics,
            }
        )
        if progress is not None:
            progress(candidate_number, len(candidates), candidate)
    return CandidateEvaluation(
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
    )


def select_top_candidates(
    evaluation: CandidateEvaluation,
    candidates: list[Candidate],
    top_k: int,
    branch_columns: tuple[str, ...] = ("model", "target_transform", "family"),
) -> tuple[list[Candidate], pd.DataFrame]:
    summary = evaluation.summary.copy()
    summary["selected"] = False
    summary["branch_rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["selection_reason"] = summary["reason"]
    eligible = summary.loc[summary["status"].eq("ok")].copy()
    if eligible.empty:
        raise RuntimeError("정상적으로 완료된 CV 후보가 없습니다.")
    selected_ids: set[str] = set()
    for _, branch in eligible.groupby(list(branch_columns), sort=True, dropna=False):
        ranked = branch.sort_values(
            ["cv_mape_mean", "cv_mape_std", "n_features", "candidate_id"],
            kind="stable",
        )
        summary.loc[ranked.index, "branch_rank"] = np.arange(1, len(ranked) + 1)
        selected_ids.update(ranked.head(top_k)["candidate_id"])
    summary.loc[summary["candidate_id"].isin(selected_ids), "selected"] = True
    summary.loc[
        summary["selected"], "selection_reason"
    ] = f"분기별 CV 상위 {top_k}개"
    summary.loc[
        summary["status"].eq("ok") & ~summary["selected"], "selection_reason"
    ] = "분기 내 CV 순위로 탈락"
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    return [candidate_map[candidate_id] for candidate_id in sorted(selected_ids)], summary


def fit_candidate(
    candidate: Candidate,
    feature_set: FeatureSet,
    feature_table: pd.DataFrame,
    train_ids: tuple[str, ...],
    all_feature_columns: tuple[str, ...],
    config: ExperimentConfig,
):
    train_table = feature_table.loc[feature_table["cell_id"].isin(train_ids)]
    estimator = build_pipeline(candidate, feature_set, all_feature_columns, config)
    estimator.fit(
        train_table.loc[:, list(all_feature_columns)], train_table["cycle_life"]
    )
    return estimator


def evaluate_fixed_model(
    estimator,
    candidate: Candidate,
    feature_table: pd.DataFrame,
    cell_ids: tuple[str, ...],
    all_feature_columns: tuple[str, ...],
    dataset_name: str,
) -> tuple[dict, pd.DataFrame]:
    table = (
        feature_table.loc[feature_table["cell_id"].isin(cell_ids)]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    predicted = estimator.predict(table.loc[:, list(all_feature_columns)])
    metrics = regression_metrics(table["cycle_life"], predicted)
    predictions = table[
        ["cell_id", "batch", "charging_policy", "cycle_life"]
    ].rename(columns={"cycle_life": "actual"})
    predictions["predicted"] = predicted
    predictions["signed_error"] = predictions["predicted"] - predictions["actual"]
    predictions["absolute_error"] = predictions["signed_error"].abs()
    predictions["ape_pct"] = predictions["absolute_error"] / predictions["actual"] * 100
    predictions["dataset"] = dataset_name
    predictions["candidate_id"] = candidate.candidate_id
    return {"dataset": dataset_name, "n": len(table), **metrics}, predictions


def evaluate_dummy_baseline(
    feature_table: pd.DataFrame,
    split: SplitBundle,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    train = feature_table.loc[feature_table["cell_id"].isin(split.train_ids)].copy()
    fold_map = split.cv_folds.set_index("cell_id")["validation_fold"]
    train["validation_fold"] = train["cell_id"].map(fold_map)
    fold_metrics = []
    for fold_number in sorted(train["validation_fold"].unique()):
        fit = train.loc[train["validation_fold"].ne(fold_number)]
        validation = train.loc[train["validation_fold"].eq(fold_number)]
        prediction = np.repeat(fit["cycle_life"].mean(), len(validation))
        fold_metrics.append(regression_metrics(validation["cycle_life"], prediction))
    fold_frame = pd.DataFrame(fold_metrics)
    model = DummyRegressor(strategy="mean").fit(
        np.ones((len(train), 1)), train["cycle_life"]
    )
    rows = [
        {
            "dataset": "Train CV",
            "n": len(train),
            "mape_pct": fold_frame["mape_pct"].mean(),
            "mae_cycles": fold_frame["mae_cycles"].mean(),
            "rmse_cycles": fold_frame["rmse_cycles"].mean(),
        }
    ]
    prediction_tables: dict[str, pd.DataFrame] = {}
    for name, ids in (("Valid", split.valid_ids), ("Batch2", split.test_ids)):
        table = feature_table.loc[feature_table["cell_id"].isin(ids)].sort_values("cell_id")
        predicted = model.predict(np.ones((len(table), 1)))
        metrics = regression_metrics(table["cycle_life"], predicted)
        rows.append({"dataset": name, "n": len(table), **metrics})
        predictions = table[["cell_id", "batch", "charging_policy", "cycle_life"]].rename(
            columns={"cycle_life": "actual"}
        )
        predictions["predicted"] = predicted
        predictions["signed_error"] = predictions["predicted"] - predictions["actual"]
        predictions["absolute_error"] = predictions["signed_error"].abs()
        predictions["ape_pct"] = predictions["absolute_error"] / predictions["actual"] * 100
        prediction_tables[name] = predictions
    return pd.DataFrame(rows), prediction_tables
