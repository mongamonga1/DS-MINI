"""모델 선택과 분리된 공통 EDA 및 설명용 진단."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ExperimentConfig
from .data import DataBundle, SplitBundle


def longest_zero_current_duration(
    time_values: np.ndarray | None,
    current_values: np.ndarray | None,
    threshold_amp: float,
) -> float:
    if time_values is None or current_values is None:
        return np.nan
    time_array = np.asarray(time_values, dtype=float)
    current_array = np.asarray(current_values, dtype=float)
    if time_array.shape != current_array.shape or time_array.size < 2:
        return np.nan
    valid = np.isfinite(time_array) & np.isfinite(current_array)
    zero_current = valid & (np.abs(current_array) < threshold_amp)
    longest = 0.0
    start_index: int | None = None
    for index, is_zero in enumerate(zero_current):
        if is_zero and start_index is None:
            start_index = index
        if not is_zero and start_index is not None:
            end_index = index - 1
            duration = time_array[end_index] - time_array[start_index]
            if np.isfinite(duration) and duration >= 0:
                longest = max(longest, float(duration))
            start_index = None
    if start_index is not None:
        duration = time_array[-1] - time_array[start_index]
        if np.isfinite(duration) and duration >= 0:
            longest = max(longest, float(duration))
    return longest


def _protocol_variant(policy: str) -> str:
    normalized = str(policy).lower()
    if "newstructure" in normalized:
        return "newstructure"
    if normalized == "unknown":
        return "unknown"
    return "기존 표기"


def _life_distribution(feature_table: pd.DataFrame, split: SplitBundle) -> pd.DataFrame:
    roles = (
        split.assignments[["cell_id", "role"]]
        .drop_duplicates("cell_id")
        .set_index("cell_id")["role"]
    )
    labelled = feature_table.loc[feature_table["cycle_life"].notna()].copy()
    labelled["role"] = labelled["cell_id"].map(roles)
    role_names = {"train": "Train", "valid": "Valid", "test": "Batch2"}
    rows = []
    for role, display_name in [("batch1", "Batch1"), *role_names.items()]:
        values = labelled.loc[
            labelled["batch"].eq("Batch 1") if role == "batch1" else labelled["role"].eq(role),
            "cycle_life",
        ]
        short_count = int(values.lt(500).sum())
        rows.append(
            {
                "dataset": display_name,
                "n": len(values),
                "short_life_count": short_count,
                "short_life_pct": short_count / len(values) * 100 if len(values) else np.nan,
                "long_life_count": int(values.gt(1000).sum()),
                "long_life_pct": values.gt(1000).mean() * 100,
                "minimum": values.min(),
                "median": values.median(),
                "mean": values.mean(),
                "maximum": values.max(),
            }
        )
    return pd.DataFrame(rows)


def _capacity_behavior(
    feature_table: pd.DataFrame, split: SplitBundle
) -> pd.DataFrame:
    train = feature_table.loc[feature_table["cell_id"].isin(split.train_ids)]
    slope = train["qd_slope"].dropna()
    relative = train["relative_capacity_change"].dropna()
    return pd.DataFrame(
        [
            {
                "dataset": "Train",
                "n": len(train),
                "valid_qd_slope": len(slope),
                "nondecreasing_qd_slope_count": int(slope.ge(0).sum()),
                "nondecreasing_qd_slope_pct": slope.ge(0).mean() * 100,
                "valid_relative_change": len(relative),
                "nonnegative_relative_change_count": int(relative.ge(0).sum()),
                "nonnegative_relative_change_pct": relative.ge(0).mean() * 100,
                "median_qd_slope": slope.median(),
                "median_relative_change": relative.median(),
            }
        ]
    )


def _train_correlations(
    feature_table: pd.DataFrame,
    split: SplitBundle,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = feature_table.loc[feature_table["cell_id"].isin(split.train_ids)]
    correlation_rows = []
    for feature in feature_columns:
        pair = train[[feature, "cycle_life"]].dropna()
        if len(pair) < 3 or pair[feature].nunique() < 2:
            continue
        correlation_rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "pearson": pair[feature].corr(pair["cycle_life"], method="pearson"),
                "spearman": pair[feature].corr(pair["cycle_life"], method="spearman"),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    if not correlations.empty:
        correlations["absolute_spearman"] = correlations["spearman"].abs()
        correlations = correlations.sort_values(
            ["absolute_spearman", "feature"], ascending=[False, True], kind="stable"
        )

    redundancy_rows = []
    for left_index, left_feature in enumerate(feature_columns):
        for right_feature in feature_columns[left_index + 1 :]:
            pair = train[[left_feature, right_feature]].dropna()
            if (
                len(pair) < 3
                or pair[left_feature].nunique() < 2
                or pair[right_feature].nunique() < 2
            ):
                continue
            spearman = pair[left_feature].corr(pair[right_feature], method="spearman")
            if np.isfinite(spearman) and abs(spearman) >= 0.9:
                redundancy_rows.append(
                    {
                        "feature_a": left_feature,
                        "feature_b": right_feature,
                        "n": len(pair),
                        "spearman": spearman,
                        "absolute_spearman": abs(spearman),
                    }
                )
    redundant = pd.DataFrame(redundancy_rows)
    if not redundant.empty:
        redundant = redundant.sort_values(
            ["absolute_spearman", "feature_a", "feature_b"],
            ascending=[False, True, True],
            kind="stable",
        )
    return correlations, redundant


def _protocol_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    labelled = feature_table.loc[feature_table["cycle_life"].notna()].copy()
    labelled["protocol_variant"] = labelled["charging_policy"].map(_protocol_variant)
    return (
        labelled.groupby(
            ["batch", "charging_policy", "protocol_variant"],
            observed=True,
            dropna=False,
        )["cycle_life"]
        .agg(n="size", mean_life="mean", median_life="median", minimum="min", maximum="max")
        .reset_index()
        .sort_values(["batch", "mean_life", "charging_policy"], kind="stable")
    )


def _rest_tables(
    data: DataBundle, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cell in data.cells:
        duration = longest_zero_current_duration(
            cell.cycle5_time,
            cell.cycle5_current,
            config.rest_current_threshold_amp,
        )
        if not np.isfinite(duration):
            rest_group = "확인 불가"
        elif duration > config.long_rest_threshold_minutes:
            rest_group = "긴 무전류 구간"
        else:
            rest_group = "짧은 무전류 구간"
        rows.append(
            {
                "cell_id": cell.cell_id,
                "batch": cell.batch,
                "cycle_life": cell.cycle_life,
                "charging_policy": cell.charging_policy,
                "protocol_variant": _protocol_variant(cell.charging_policy),
                "cycle5_longest_zero_current_minutes": duration,
                "rest_group": rest_group,
            }
        )
    cells = pd.DataFrame(rows)
    summary_rows = []
    for (batch, rest_group), group in cells.groupby(
        ["batch", "rest_group"], observed=True, sort=False
    ):
        labelled = group.loc[group["cycle_life"].notna()]
        summary_rows.append(
            {
                "batch": batch,
                "rest_group": rest_group,
                "all_cells": len(group),
                "labelled_cells": len(labelled),
                "minimum_rest_minutes": group[
                    "cycle5_longest_zero_current_minutes"
                ].min(),
                "maximum_rest_minutes": group[
                    "cycle5_longest_zero_current_minutes"
                ].max(),
                "mean_life": labelled["cycle_life"].mean(),
                "short_life_count": int(labelled["cycle_life"].lt(500).sum()),
            }
        )
    return cells, pd.DataFrame(summary_rows)


def _full_qd_trajectories(data: DataBundle, split: SplitBundle) -> pd.DataFrame:
    role_map = split.assignments.set_index("cell_id")["role"]
    frames = []
    for cell in data.cells:
        trajectory = cell.full_qd.copy()
        trajectory["cell_id"] = cell.cell_id
        trajectory["batch"] = cell.batch
        trajectory["role"] = role_map.get(cell.cell_id, "unknown")
        trajectory["cycle_life"] = cell.cycle_life
        frames.append(trajectory)
    return pd.concat(frames, ignore_index=True)


def build_eda_tables(
    data: DataBundle,
    feature_table: pd.DataFrame,
    split: SplitBundle,
    feature_columns: tuple[str, ...],
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    correlations, redundant = _train_correlations(
        feature_table, split, feature_columns
    )
    rest_cells, rest_summary = _rest_tables(data, config)
    role_map = split.assignments.set_index("cell_id")["role"]
    life_cells = feature_table[
        ["cell_id", "batch", "cycle_life", "charging_policy"]
    ].copy()
    life_cells["role"] = life_cells["cell_id"].map(role_map)
    train_feature_values = feature_table.loc[
        feature_table["cell_id"].isin(split.train_ids),
        [
            "cell_id",
            "cycle_life",
            "mean_qd",
            "qd_slope",
            "relative_capacity_change",
            "delta_q_log_variance",
            "delta_q_min",
        ],
    ].copy()
    extra = _additional_tables(data, feature_table, split, feature_columns)
    return {
        **extra,
        "eda_settings": pd.DataFrame([{"hist_min": config.life_histogram_range[0], "hist_max": config.life_histogram_range[1], "show_plots": config.show_eda_plots}]),
        "life_distribution": _life_distribution(feature_table, split),
        "life_cells": life_cells,
        "capacity_behavior": _capacity_behavior(feature_table, split),
        "train_feature_values": train_feature_values,
        "train_feature_correlations": correlations,
        "train_redundant_pairs": redundant,
        "protocol_summary": _protocol_summary(feature_table),
        "rest_cells": rest_cells,
        "rest_summary": rest_summary,
        "rest_config": pd.DataFrame(
            [
                {
                    "current_threshold_amp": config.rest_current_threshold_amp,
                    "long_rest_threshold_minutes": config.long_rest_threshold_minutes,
                }
            ]
        ),
        "full_qd_trajectories": _full_qd_trajectories(data, split),
    }


def _best_mape(table: pd.DataFrame, feature_set: str) -> float:
    if table.empty:
        return np.nan
    matched = table.loc[
        table["status"].eq("ok") & table["feature_set"].eq(feature_set),
        "cv_mape_mean",
    ]
    return float(matched.min()) if not matched.empty else np.nan


def build_eda_findings(
    tables: dict[str, pd.DataFrame],
    stage_results: dict[str, pd.DataFrame],
    final_configurations: pd.DataFrame,
    error_analysis: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    life = tables["life_distribution"].set_index("dataset")
    train_short = int(life.loc["Train", "short_life_count"])
    test_short = int(life.loc["Batch2", "short_life_count"])
    test_short_pct = float(life.loc["Batch2", "short_life_pct"])
    capacity = tables["capacity_behavior"].iloc[0]
    protocol = tables["protocol_summary"]
    rest = tables["rest_summary"]
    correlations = tables["train_feature_correlations"]
    redundant = tables["train_redundant_pairs"]

    if error_analysis:
        short_error = error_analysis["short_life_summary"].iloc[0]
        short_evidence = (
            f"Batch2 단수명 {int(short_error['n_short_life'])}셀 중 "
            f"{int(short_error['overprediction_count'])}셀을 과대 예측했고 "
            f"MAPE는 {short_error['mape_pct']:.1f}%였다."
        )
    else:
        short_evidence = "평가 전이며, 최종 모델 고정 후 단수명 과대 예측을 확인한다."

    stage1 = stage_results.get("stage1", pd.DataFrame())
    policy_mape = _best_mape(stage1, "C_policy")
    baseline_b_mape = _best_mape(stage1, "B")
    if np.isfinite(policy_mape) and np.isfinite(baseline_b_mape):
        policy_evidence = (
            f"1차 CV 최적 C-policy MAPE {policy_mape:.2f}%, "
            f"B 계열 {baseline_b_mape:.2f}%로 숫자 정책 피처의 추가 이득은 제한적이었다."
        )
    else:
        policy_evidence = "평가 전이며 C-policy와 B 후보를 같은 Train CV로 비교한다."

    if not final_configurations.empty:
        existing = final_configurations.loc[
            final_configurations["configuration"].eq("existing_representative")
        ]
        changed = final_configurations.loc[
            final_configurations["configuration"].eq("changed_representative")
        ]
        if not existing.empty and not changed.empty:
            capacity_evidence = (
                f"변경 피처 대표 CV MAPE {changed.iloc[0]['cv_mape_mean']:.2f}%와 "
                f"기존 피처 대표 {existing.iloc[0]['cv_mape_mean']:.2f}%를 비교했다."
            )
        else:
            capacity_evidence = "최종 대표 구성에서 기존·변경 피처를 비교했다."
    else:
        capacity_evidence = "평가 전이며 기존·변경 피처 대표를 Train CV로 비교한다."

    batch2_rest = rest.loc[rest["batch"].eq("Batch 2")]
    rest_observation = "; ".join(
        f"{row.rest_group} {int(row.labelled_cells)}셀, 평균수명 {row.mean_life:.1f}"
        for row in batch2_rest.itertuples()
        if row.labelled_cells
    )
    if error_analysis:
        rest_errors = error_analysis["group_metrics"].loc[
            error_analysis["group_metrics"]["group_type"].eq("rest_group")
        ]
        rest_evidence = "; ".join(
            f"{row.group} MAPE {row.mape_pct:.1f}% (n={int(row.n)})"
            for row in rest_errors.itertuples()
        )
    else:
        rest_evidence = "모델 고정 후 무전류 그룹별 오류를 사후 비교한다."

    top_correlation = (
        f"{correlations.iloc[0]['feature']}의 Spearman "
        f"{correlations.iloc[0]['spearman']:.3f}"
        if not correlations.empty
        else "유효 상관 없음"
    )
    model_evidence = (
        f"최종 모델은 {final_configurations.iloc[0]['model']}이며 "
        f"타깃 변환은 {final_configurations.iloc[0]['target_transform']}였다."
        if not final_configurations.empty
        else "규제 선형모델과 Random Forest를 같은 Train CV로 비교한다."
    )

    findings = pd.DataFrame(
        [
            {
                "주제": "단수명 분포 이동",
                "관측한 내용": f"Train 단수명 {train_short}셀, Batch2는 {test_short}셀({test_short_pct:.1f}%).",
                "예측 문제": "학습 범위보다 짧은 셀의 수명을 길게 예측할 위험이 있다.",
                "대응 후보": "원·로그 타깃과 제한된 Random Forest를 비교하고 단수명 셀을 제거하지 않았다.",
                "CV·평가 확인": short_evidence,
            },
            {
                "주제": "초기 용량 변화",
                "관측한 내용": (
                    f"Train에서 Qd 기울기가 0 이상인 셀은 "
                    f"{int(capacity['nondecreasing_qd_slope_count'])}/{int(capacity['valid_qd_slope'])}셀이다."
                ),
                "예측 문제": "초기 평균 용량이나 단일 기울기만으로 열화를 구분하기 어렵다.",
                "대응 후보": "ΔQ A·B, 평균 Qd 제외, 상대 용량 변화, 정규화 ΔQ 후보를 비교했다.",
                "CV·평가 확인": capacity_evidence,
            },
            {
                "주제": "충전 프로토콜 표기",
                "관측한 내용": f"원문 기준 프로토콜 {len(protocol)}개 조합을 확인했고 접미사를 보존했다.",
                "예측 문제": "같은 C1·C2·SOC라도 newstructure 등 실험 조건 차이를 숫자 세 개가 모두 담지 못한다.",
                "대응 후보": "C-policy 후보를 비교하되 원문·접미사는 모델 입력이 아닌 사후 그룹 분석에 유지했다.",
                "CV·평가 확인": policy_evidence,
            },
            {
                "주제": "cycle5 무전류 구간",
                "관측한 내용": rest_observation or "유효한 cycle5 전류·시간 기록이 없다.",
                "예측 문제": "배치·실험 조건 이동이 단수명 과대 예측과 함께 나타날 수 있다.",
                "대응 후보": "무전류 그룹은 설명·오류 분석에만 사용하고 모델 선택에는 넣지 않았다.",
                "CV·평가 확인": rest_evidence or "유효한 사후 오류 그룹이 없다.",
            },
            {
                "주제": "Train 상관·중복",
                "관측한 내용": f"Train 전용 상위 연관은 {top_correlation}, |Spearman|≥0.9 피처쌍은 {len(redundant)}개다.",
                "예측 문제": "소표본에서 상관 탐색과 중복 피처가 계수와 선택을 불안정하게 만들 수 있다.",
                "대응 후보": "피처군을 나누고 Ridge·ElasticNet 규제와 제한된 후보 선별을 사용했다.",
                "CV·평가 확인": model_evidence,
            },
        ]
    )
    transform = tables["train_transform_values"]
    extra = pd.DataFrame([
        {"주제": "ΔQ 곡선", "관측한 내용": f"전압별 Q100-Q10을 {tables['delta_q_curves'].cell_id.nunique()}셀에서 확인했다.", "예측 문제": "총용량 평균에서 보이지 않는 전압별 변화를 요약해야 한다.", "대응 후보": "ΔQ 로그 분산·최솟값 A/B와 정규화 표현을 비교한다.", "CV·평가 확인": capacity_evidence},
        {"주제": "원본·로그 변환", "관측한 내용": f"Train 수명 왜도 {transform.cycle_life.skew():.2f}, log10 수명 왜도 {transform.log10_cycle_life.skew():.2f}.", "예측 문제": "분포 압축 자체가 예측 성능을 보장하지 않는다.", "대응 후보": "원 수명·log10 수명을 같은 CV에서 비교한다.", "CV·평가 확인": model_evidence},
        {"주제": "배치 피처·프로토콜 차이", "관측한 내용": "Train P5~P95 범위와 배치별 주요 피처 분포·미관측 프로토콜 수를 확인했다.", "예측 문제": "다른 배치에서 입력 범위와 조건이 바뀔 수 있다.", "대응 후보": "Batch2 분포를 유지하고 피처를 자동 삭제·추가하지 않는다.", "CV·평가 확인": short_evidence},
    ])
    return pd.concat([findings, extra], ignore_index=True)


def _additional_tables(data, features, split, columns):
    """설명용 곡선·분포를 계산하며 모델 피처 표는 변경하지 않습니다."""
    train = features.loc[features.cell_id.isin(split.train_ids)].copy()
    train["delta_q_variance"] = np.power(10.0, train.delta_q_log_variance)
    train["log10_cycle_life"] = np.log10(train.cycle_life)
    # 원분산은 로그 하한 적용 전의 곡선으로 재계산합니다.
    curves = []
    for cell in data.cells:
        if cell.q10 is None or cell.q100 is None:
            continue
        if not (len(cell.voltage) == len(cell.q10) == len(cell.q100)):
            continue
        valid = np.isfinite(cell.voltage) & np.isfinite(cell.q10) & np.isfinite(cell.q100)
        if valid.sum() < 2:
            continue
        delta = cell.q100[valid] - cell.q10[valid]
        train.loc[train.cell_id.eq(cell.cell_id), "delta_q_variance"] = np.var(delta, ddof=0)
        curve = pd.DataFrame({"voltage": cell.voltage[valid], "delta_q": delta})
        curve["cell_id"] = cell.cell_id
        curve["batch"] = cell.batch
        curve["life_group"] = "unlabelled" if not np.isfinite(cell.cycle_life) else "<500" if cell.cycle_life < 500 else ">1000" if cell.cycle_life > 1000 else "500-1000"
        curves.append(curve.sort_values("voltage"))
    curve_table = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["voltage", "delta_q", "cell_id", "batch", "life_group"])
    delta_values = features[["cell_id", "batch", "cycle_life", "delta_q_log_variance", "delta_q_min"]].dropna(subset=["cycle_life"]).copy()
    delta_values["life_group"] = np.select([delta_values.cycle_life.lt(500), delta_values.cycle_life.gt(1000)], ["<500", ">1000"], default="500-1000")
    policy_correlations = []
    for batch, group in features.dropna(subset=["cycle_life"]).groupby("batch"):
        for column in ["c_rate_stage1", "c_rate_stage2", "switch_soc_pct"]:
            pair = group[[column, "cycle_life"]].dropna()
            policy_correlations.append({"batch": batch, "feature": column, "n": len(pair), "spearman": pair[column].corr(pair.cycle_life, method="spearman") if pair[column].nunique() > 1 else np.nan, "scope": "설명용·후보 선택에 사용하지 않음"})
    ranges = []
    for column in columns:
        values = train[column].dropna()
        low, high = values.quantile([0.05, 0.95]) if len(values) else (np.nan, np.nan)
        for batch, group in features.groupby("batch"):
            finite = group[column].dropna()
            ranges.append({"batch": batch, "feature": column, "n": len(finite), "median": finite.median(), "train_p05": low, "train_p95": high, "outside_train_p05_p95_pct": ((finite < low) | (finite > high)).mean() * 100 if len(values) and len(finite) else np.nan})
    protocols = set(train.charging_policy)
    overlap = features.groupby("batch").agg(all_cells=("cell_id", "size"))
    overlap["unseen_protocol_cells"] = features.assign(unseen=~features.charging_policy.isin(protocols)).groupby("batch").unseen.sum()
    return {
        "delta_q_curves": curve_table,
        "delta_q_groups": delta_values,
        "train_transform_values": train[["cell_id", "cycle_life", "log10_cycle_life", "delta_q_variance", "delta_q_log_variance"]],
        "train_feature_correlation_matrix": train[list(columns)].corr(method="spearman"),
        "policy_correlations": pd.DataFrame(policy_correlations),
        "batch_feature_distribution": features[["cell_id", "batch", *columns]].copy(),
        "batch_feature_coverage": pd.DataFrame(ranges),
        "protocol_coverage": overlap.reset_index(),
    }
