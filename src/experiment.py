"""데이터 준비부터 최종 평가까지 공통 실행 순서."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import platform

import numpy as np
import pandas as pd

from config import ExperimentConfig
from .data import DataBundle, SplitBundle, load_battery_data, make_cell_splits
from .eda import build_eda_tables
from .features import (
    FeatureSet,
    all_feature_columns,
    build_feature_table,
    expand_feature_set,
    initial_feature_sets,
)
from .modeling import (
    Candidate,
    CandidateEvaluation,
    build_initial_candidates,
    build_tuned_candidates,
    evaluate_candidates,
    evaluate_dummy_baseline,
    evaluate_fixed_model,
    fit_candidate,
    make_candidate,
    regression_metrics,
    select_top_candidates,
)


@dataclass
class ExperimentResult:
    environment: pd.DataFrame
    settings: pd.DataFrame
    data_quality: pd.DataFrame
    feature_quality: pd.DataFrame
    split_assignments: pd.DataFrame
    cv_folds: pd.DataFrame
    feature_catalog: pd.DataFrame
    eda_tables: dict[str, pd.DataFrame]
    stage_results: dict[str, pd.DataFrame]
    selection_history: dict[str, pd.DataFrame]
    final_configurations: pd.DataFrame
    model_performance: pd.DataFrame
    gaps: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    error_analysis: dict[str, pd.DataFrame]
    selected_models: dict[str, object]


def _environment_table() -> pd.DataFrame:
    rows = [{"항목": "Python", "버전": platform.python_version()}]
    for package in ["numpy", "pandas", "h5py", "scikit-learn", "matplotlib", "joblib"]:
        rows.append({"항목": package, "버전": importlib.metadata.version(package)})
    return pd.DataFrame(rows)


def _settings_table(config: ExperimentConfig, data_dir) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"설정": "데이터 위치", "값": str(data_dir)},
            {"설정": "난수", "값": config.random_state},
            {"설정": "Batch1 검증 셀 수", "값": config.valid_size},
            {"설정": "Train CV", "값": f"{config.cv_folds}-fold"},
            {"설정": "타깃 후보", "값": ", ".join(config.target_transforms)},
            {"설정": "모델 후보", "값": ", ".join(config.model_names)},
            {"설정": "결측 대체", "값": "CV 학습 fold 중앙값"},
            {"설정": "표준화", "값": "CV 학습 fold 평균·표준편차"},
            {"설정": "주 선택 지표", "값": "CV 평균 MAPE"},
            {"설정": "1차 분기별 생존 수", "값": config.top_k_stage1},
            {"설정": "확장 분기별 생존 수", "값": config.top_k_stage2},
            {
                "설정": "무전류 기준",
                "값": f"|I| < {config.rest_current_threshold_amp} A",
            },
            {
                "설정": "긴 무전류 구간",
                "값": f"> {config.long_rest_threshold_minutes}분",
            },
            {"설정": "최종 평가", "값": config.run_final_evaluation},
        ]
    )


def _catalog(feature_sets: dict[str, FeatureSet]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": item.name,
                "family": item.family,
                "variant": item.variant,
                "parent": item.parent or "",
                "n_features": len(item.columns),
                "features": ", ".join(item.columns),
            }
            for item in feature_sets.values()
        ]
    ).sort_values(["family", "feature_set"], kind="stable")


def _combine_evaluations(*evaluations: CandidateEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        summary=pd.concat([item.summary for item in evaluations], ignore_index=True),
        fold_metrics=pd.concat(
            [item.fold_metrics for item in evaluations if not item.fold_metrics.empty],
            ignore_index=True,
        ),
        oof_predictions=pd.concat(
            [item.oof_predictions for item in evaluations if not item.oof_predictions.empty],
            ignore_index=True,
        ),
    )


def _expanded_candidates(
    survivors: list[Candidate],
    feature_sets: dict[str, FeatureSet],
    config: ExperimentConfig,
) -> tuple[list[Candidate], dict[str, FeatureSet]]:
    expanded_sets: dict[str, FeatureSet] = {}
    candidates: dict[str, Candidate] = {}
    for survivor in survivors:
        for name, expanded in expand_feature_set(feature_sets[survivor.feature_set]).items():
            expanded_sets[name] = expanded
            candidate = make_candidate(
                "stage2_expansion",
                expanded,
                survivor.model_name,
                survivor.target_transform,
                config.default_model_params[survivor.model_name],
            )
            candidates[candidate.candidate_id] = candidate
    return list(candidates.values()), expanded_sets


def _candidate_lookup(candidates: list[Candidate]) -> dict[str, Candidate]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def _rank_final(evaluation: CandidateEvaluation) -> pd.DataFrame:
    eligible = evaluation.summary.loc[evaluation.summary["status"].eq("ok")].copy()
    if eligible.empty:
        raise RuntimeError("매개변수 탐색을 정상 완료한 후보가 없습니다.")
    return eligible.sort_values(
        ["cv_mape_mean", "cv_mape_std", "n_features", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def _representatives(
    ranked: pd.DataFrame, candidate_map: dict[str, Candidate]
) -> dict[str, Candidate]:
    selected: dict[str, Candidate] = {"final": candidate_map[ranked.iloc[0]["candidate_id"]]}
    for variant, label in (("existing", "existing_representative"), ("changed", "changed_representative")):
        rows = ranked.loc[ranked["variant"].eq(variant)]
        if not rows.empty:
            selected[label] = candidate_map[rows.iloc[0]["candidate_id"]]
    unique: dict[str, Candidate] = {}
    for label, candidate in selected.items():
        unique[label] = candidate
    return unique


def _performance_and_predictions(
    representatives: dict[str, Candidate],
    stage3: CandidateEvaluation,
    feature_sets: dict[str, FeatureSet],
    feature_table: pd.DataFrame,
    split: SplitBundle,
    columns: tuple[str, ...],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    stage3_index = stage3.summary.set_index("candidate_id")
    performance_rows: list[dict] = []
    prediction_tables: dict[str, pd.DataFrame] = {}
    models: dict[str, object] = {}
    for label, candidate in representatives.items():
        summary = stage3_index.loc[candidate.candidate_id]
        performance_rows.append(
            {
                "configuration": label,
                "candidate_id": candidate.candidate_id,
                "dataset": "Train CV",
                "n": 36,
                "mape_pct": summary["cv_mape_mean"],
                "mae_cycles": summary["cv_mae_mean"],
                "rmse_cycles": summary["cv_rmse_mean"],
            }
        )
        estimator = fit_candidate(
            candidate,
            feature_sets[candidate.feature_set],
            feature_table,
            split.train_ids,
            columns,
            config,
        )
        models[label] = estimator
        for dataset_name, ids in (("Valid", split.valid_ids), ("Batch2", split.test_ids)):
            metrics, predictions = evaluate_fixed_model(
                estimator,
                candidate,
                feature_table,
                ids,
                columns,
                dataset_name,
            )
            performance_rows.append(
                {
                    "configuration": label,
                    "candidate_id": candidate.candidate_id,
                    **metrics,
                }
            )
            prediction_tables[f"{label}:{dataset_name}"] = predictions

    baseline_performance, baseline_predictions = evaluate_dummy_baseline(feature_table, split)
    baseline_performance.insert(0, "configuration", "dummy_mean")
    baseline_performance.insert(1, "candidate_id", "dummy_mean")
    performance = pd.concat(
        [baseline_performance, pd.DataFrame(performance_rows)], ignore_index=True
    )
    for dataset_name, predictions in baseline_predictions.items():
        prediction_tables[f"dummy_mean:{dataset_name}"] = predictions
    return performance, prediction_tables, models


def _gap_table(performance: pd.DataFrame, target_mape_pct: float) -> pd.DataFrame:
    rows: list[dict] = []
    for configuration, group in performance.groupby("configuration", sort=False):
        values = group.set_index("dataset")["mape_pct"]
        if not {"Train CV", "Valid", "Batch2"}.issubset(values.index):
            continue
        rows.extend(
            [
                {
                    "configuration": configuration,
                    "gap": "Valid - CV",
                    "value_pct_point": values["Valid"] - values["Train CV"],
                },
                {
                    "configuration": configuration,
                    "gap": "Test - Valid",
                    "value_pct_point": values["Batch2"] - values["Valid"],
                },
                {
                    "configuration": configuration,
                    "gap": f"Test - {target_mape_pct:g}",
                    "value_pct_point": values["Batch2"] - target_mape_pct,
                },
            ]
        )
    return pd.DataFrame(rows)


def _error_analysis(
    final_predictions: pd.DataFrame,
    feature_table: pd.DataFrame,
    split: SplitBundle,
    columns: tuple[str, ...],
    eda_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    predictions = final_predictions.copy()
    explanatory = eda_tables["rest_cells"][
        [
            "cell_id",
            "protocol_variant",
            "cycle5_longest_zero_current_minutes",
            "rest_group",
        ]
    ]
    predictions = predictions.merge(
        explanatory, on="cell_id", how="left", validate="one_to_one"
    )
    predictions["life_group"] = np.select(
        [predictions["actual"].lt(500), predictions["actual"].gt(1000)],
        ["<500", ">1000"], default="500-1000",
    )
    group_rows: list[dict] = []
    for group_column in (
        "life_group",
        "charging_policy",
        "protocol_variant",
        "rest_group",
    ):
        for group_name, group in predictions.groupby(group_column, observed=True, sort=False):
            group_rows.append(
                {
                    "group_type": group_column,
                    "group": str(group_name),
                    "n": len(group),
                    **regression_metrics(group["actual"], group["predicted"]),
                    "mean_signed_error": group["signed_error"].mean(),
                    "overprediction_count": int(group["signed_error"].gt(0).sum()),
                }
            )
    short = predictions.loc[predictions["actual"].lt(500)]
    short_summary = pd.DataFrame(
        [
            {
                "n_short_life": len(short),
                "overprediction_count": int(short["signed_error"].gt(0).sum()),
                "mean_signed_error": short["signed_error"].mean(),
                "mape_pct": (
                    np.mean(short["absolute_error"] / short["actual"]) * 100
                    if len(short)
                    else np.nan
                ),
            }
        ]
    )

    train = feature_table.loc[feature_table["cell_id"].isin(split.train_ids)]
    test = feature_table.loc[feature_table["cell_id"].isin(split.test_ids)]
    range_rows = []
    for column in columns:
        train_values = train[column].dropna()
        test_values = test[column].dropna()
        if train_values.empty:
            range_rows.append(
                {
                    "feature": column,
                    "train_min": np.nan,
                    "train_max": np.nan,
                    "test_outside_count": np.nan,
                    "test_valid_count": len(test_values),
                }
            )
            continue
        outside = test_values.lt(train_values.min()) | test_values.gt(train_values.max())
        range_rows.append(
            {
                "feature": column,
                "train_min": train_values.min(),
                "train_max": train_values.max(),
                "test_outside_count": int(outside.sum()),
                "test_valid_count": len(test_values),
            }
        )
    return {
        "group_metrics": pd.DataFrame(group_rows),
        "short_life_summary": short_summary,
        "worst_absolute": predictions.nlargest(10, "absolute_error"),
        "worst_relative": predictions.nlargest(10, "ape_pct"),
        "feature_range": pd.DataFrame(range_rows),
    }


class ExperimentRunner:
    """노트북의 단계 실행과 main.py의 일괄 실행이 공유하는 상태 객체."""

    def __init__(self, config: ExperimentConfig, *, progress: bool = True):
        config.validate()
        self.config = config
        self.progress = progress
        self.data: DataBundle | None = None
        self.split: SplitBundle | None = None
        self.feature_table: pd.DataFrame | None = None
        self.feature_quality = pd.DataFrame()
        self.eda_tables: dict[str, pd.DataFrame] = {}
        self.base_feature_sets: dict[str, FeatureSet] = {}
        self.all_feature_sets: dict[str, FeatureSet] = {}
        self.columns: tuple[str, ...] = ()
        self.stage1: CandidateEvaluation | None = None
        self.stage2: CandidateEvaluation | None = None
        self.stage3: CandidateEvaluation | None = None
        self.stage1_candidates: list[Candidate] = []
        self.stage2_candidates: list[Candidate] = []
        self.stage3_candidates: list[Candidate] = []
        self.stage1_survivors: list[Candidate] = []
        self.stage2_survivors: list[Candidate] = []
        self.stage1_selection = pd.DataFrame()
        self.stage2_selection = pd.DataFrame()
        self.stage3_selection = pd.DataFrame()
        self.representatives: dict[str, Candidate] = {}
        self.final_configurations = pd.DataFrame()
        self.performance = pd.DataFrame()
        self.gaps = pd.DataFrame()
        self.predictions: dict[str, pd.DataFrame] = {}
        self.models: dict[str, object] = {}
        self.error_analysis: dict[str, pd.DataFrame] = {}

    def _announce(self, message: str) -> None:
        if self.progress:
            print(message, flush=True)

    def _candidate_progress(self, stage_name: str):
        def report(current: int, total: int, candidate: Candidate) -> None:
            if self.progress and (current == 1 or current == total or current % 25 == 0):
                print(
                    f"  {stage_name}: {current}/{total} - {candidate.feature_set} / "
                    f"{candidate.model_name} / {candidate.target_transform}",
                    flush=True,
                )

        return report

    def load_data_and_split(self) -> DataBundle:
        if self.data is not None:
            return self.data
        self._announce("[1/10] Batch1·2 원본 읽기와 품질 처리")
        self.data = load_battery_data(self.config)
        self.split = make_cell_splits(self.data.metadata, self.config)
        return self.data

    def build_features(self) -> pd.DataFrame:
        if self.feature_table is not None:
            return self.feature_table
        if self.data is None:
            raise RuntimeError("원본 읽기·분할 단계를 먼저 실행하세요.")
        self._announce("[2/10] 초기 100사이클 피처 계산")
        self.feature_table, self.feature_quality = build_feature_table(self.data.cells)
        self.base_feature_sets = initial_feature_sets()
        self.all_feature_sets = dict(self.base_feature_sets)
        self.columns = all_feature_columns(self.base_feature_sets)
        missing_columns = sorted(set(self.columns) - set(self.feature_table.columns))
        if missing_columns:
            raise ValueError(f"계산되지 않은 공통 피처가 있습니다: {missing_columns}")
        return self.feature_table

    def run_eda(self) -> dict[str, pd.DataFrame]:
        if self.eda_tables:
            return self.eda_tables
        if self.data is None or self.feature_table is None or self.split is None:
            raise RuntimeError("피처 계산 단계를 먼저 실행하세요.")
        self._announce("[3/10] Train 근거 EDA와 설명용 배치 진단")
        self.eda_tables = build_eda_tables(
            self.data,
            self.feature_table,
            self.split,
            self.columns,
            self.config,
        )
        return self.eda_tables

    def run_stage1(self) -> CandidateEvaluation:
        if self.stage1 is not None:
            return self.stage1
        if self.feature_table is None or self.split is None or not self.eda_tables:
            raise RuntimeError("EDA 단계를 먼저 실행하세요.")
        self._announce("[4/10] 전체 피처군 × 모델 × 타깃 1차 CV")
        self.stage1_candidates = build_initial_candidates(
            self.base_feature_sets, self.config
        )
        self.stage1 = evaluate_candidates(
            self.stage1_candidates,
            self.base_feature_sets,
            self.feature_table,
            self.split,
            self.columns,
            self.config,
            self._candidate_progress("1차 CV"),
        )
        self.stage1_survivors, self.stage1_selection = select_top_candidates(
            self.stage1, self.stage1_candidates, self.config.top_k_stage1
        )
        return self.stage1

    def run_stage2(self) -> CandidateEvaluation:
        if self.stage2 is not None:
            return self.stage2
        if self.stage1 is None or self.feature_table is None or self.split is None:
            raise RuntimeError("1차 CV 단계를 먼저 실행하세요.")
        self._announce("[5/10] 생존 구성의 확장·변화 피처 CV")
        self.stage2_candidates, expanded_sets = _expanded_candidates(
            self.stage1_survivors, self.base_feature_sets, self.config
        )
        self.all_feature_sets = {**self.base_feature_sets, **expanded_sets}
        self.stage2 = evaluate_candidates(
            self.stage2_candidates,
            self.all_feature_sets,
            self.feature_table,
            self.split,
            self.columns,
            self.config,
            self._candidate_progress("확장 CV"),
        )
        survivor_ids = [
            candidate.candidate_id for candidate in self.stage1_survivors
        ]
        combined_stage2 = _combine_evaluations(
            CandidateEvaluation(
                self.stage1.summary.loc[
                    self.stage1.summary["candidate_id"].isin(survivor_ids)
                ].copy(),
                self.stage1.fold_metrics.loc[
                    self.stage1.fold_metrics["candidate_id"].isin(survivor_ids)
                ].copy(),
                self.stage1.oof_predictions.loc[
                    self.stage1.oof_predictions["candidate_id"].isin(survivor_ids)
                ].copy(),
            ),
            self.stage2,
        )
        self.stage2_survivors, self.stage2_selection = select_top_candidates(
            combined_stage2,
            [*self.stage1_survivors, *self.stage2_candidates],
            self.config.top_k_stage2,
            branch_columns=("model", "target_transform", "family", "variant"),
        )
        return self.stage2

    def run_stage3(self) -> CandidateEvaluation:
        if self.stage3 is not None:
            return self.stage3
        if self.stage2 is None or self.feature_table is None or self.split is None:
            raise RuntimeError("확장·변화 피처 CV 단계를 먼저 실행하세요.")
        self._announce("[6/10] 생존 후보 매개변수 재탐색")
        self.stage3_candidates = build_tuned_candidates(
            self.stage2_survivors, self.all_feature_sets, self.config
        )
        self.stage3 = evaluate_candidates(
            self.stage3_candidates,
            self.all_feature_sets,
            self.feature_table,
            self.split,
            self.columns,
            self.config,
            self._candidate_progress("매개변수 CV"),
        )
        ranked = _rank_final(self.stage3)
        self.stage3_selection = self.stage3.summary.copy()
        rank_map = pd.Series(
            np.arange(1, len(ranked) + 1), index=ranked["candidate_id"]
        )
        self.stage3_selection["global_rank"] = self.stage3_selection[
            "candidate_id"
        ].map(rank_map).astype("Int64")
        self.stage3_selection["selected"] = self.stage3_selection[
            "candidate_id"
        ].eq(ranked.iloc[0]["candidate_id"])
        self.stage3_selection["selection_reason"] = np.where(
            self.stage3_selection["selected"],
            "전체 CV 평균 MAPE 최솟값",
            "최종 순위로 탈락",
        )
        candidate_map = _candidate_lookup(self.stage3_candidates)
        self.representatives = _representatives(ranked, candidate_map)
        ranked_index = ranked.set_index("candidate_id")
        rows = []
        for label, candidate in self.representatives.items():
            row = ranked_index.loc[candidate.candidate_id].to_dict()
            rows.append(
                {
                    "configuration": label,
                    "candidate_id": candidate.candidate_id,
                    **row,
                }
            )
        self.final_configurations = pd.DataFrame(rows)
        return self.stage3

    def run_final_evaluation(self) -> pd.DataFrame:
        if not self.performance.empty:
            return self.performance
        if self.stage3 is None or self.feature_table is None or self.split is None:
            raise RuntimeError("매개변수 재탐색 단계를 먼저 실행하세요.")
        self._announce("[7/10] Train 36셀 최종 학습")
        self.performance, self.predictions, self.models = _performance_and_predictions(
            self.representatives,
            self.stage3,
            self.all_feature_sets,
            self.feature_table,
            self.split,
            self.columns,
            self.config,
        )
        self._announce("[8/10] Valid 10셀·Batch2 39셀 고정 평가")
        self.gaps = _gap_table(self.performance, self.config.target_mape_pct)
        return self.performance

    def run_error_analysis(self) -> dict[str, pd.DataFrame]:
        if self.error_analysis:
            return self.error_analysis
        if self.feature_table is None or self.split is None:
            raise RuntimeError("피처 계산 단계를 먼저 실행하세요.")
        final_predictions = self.predictions.get("final:Batch2")
        if final_predictions is None:
            raise RuntimeError("최종 학습·평가 단계를 먼저 실행하세요.")
        self._announce("[9/10] 단수명·프로토콜·무전류 그룹 오류 분석")
        self.error_analysis = _error_analysis(
            final_predictions,
            self.feature_table,
            self.split,
            self.columns,
            self.eda_tables,
        )
        return self.error_analysis

    def result(self) -> ExperimentResult:
        data_dir = self.data.data_dir if self.data is not None else self.config.data_dir
        return ExperimentResult(
            environment=_environment_table(),
            settings=_settings_table(self.config, data_dir),
            data_quality=(self.data.quality if self.data is not None else pd.DataFrame()),
            feature_quality=self.feature_quality,
            split_assignments=(
                self.split.assignments if self.split is not None else pd.DataFrame()
            ),
            cv_folds=self.split.cv_folds if self.split is not None else pd.DataFrame(),
            feature_catalog=_catalog(self.all_feature_sets)
            if self.all_feature_sets
            else pd.DataFrame(),
            eda_tables=self.eda_tables,
            stage_results={
                name: evaluation.summary
                for name, evaluation in (
                    ("stage1", self.stage1),
                    ("stage2", self.stage2),
                    ("stage3", self.stage3),
                )
                if evaluation is not None
            },
            selection_history={
                name: selection
                for name, selection in (
                    ("stage1", self.stage1_selection),
                    ("stage2", self.stage2_selection),
                    ("stage3", self.stage3_selection),
                )
                if not selection.empty
            },
            final_configurations=self.final_configurations,
            model_performance=self.performance,
            gaps=self.gaps,
            predictions=self.predictions,
            error_analysis=self.error_analysis,
            selected_models=self.models,
        )

    def run_all(self) -> ExperimentResult:
        self.load_data_and_split()
        self.build_features()
        self.run_eda()
        self.run_stage1()
        self.run_stage2()
        self.run_stage3()
        if self.config.run_final_evaluation:
            self.run_final_evaluation()
            self.run_error_analysis()
        self._announce("[10/10] 공통 결과 객체 구성 완료")
        return self.result()


def run_experiment(
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> ExperimentResult:
    """main.py에서 전체 단계를 한 번에 실행합니다."""
    return ExperimentRunner(config, progress=progress).run_all()
