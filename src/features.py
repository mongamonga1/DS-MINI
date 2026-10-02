"""초기 100사이클 피처 계산과 후보 구성."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CellRecord


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]
    family: str
    variant: str
    parent: str | None = None


def _valid_mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if valid.size else np.nan


def _window_mean(summary: pd.DataFrame, column: str, start: int, end: int) -> float:
    mask = summary["cycle"].between(start, end)
    return _valid_mean(summary.loc[mask, column])


def _difference_between_windows(summary: pd.DataFrame, column: str) -> float:
    early = _window_mean(summary, column, 2, 11)
    late = _window_mean(summary, column, 91, 100)
    return float(late - early) if np.isfinite(early) and np.isfinite(late) else np.nan


def _delta_q_features(cell: CellRecord, mean_qd_early: float) -> tuple[dict, dict]:
    values = {
        "delta_q_log_variance": np.nan,
        "delta_q_min": np.nan,
        "normalized_delta_q_log_variance": np.nan,
        "normalized_delta_q_min": np.nan,
    }
    quality = {
        "delta_q_status": "정상",
        "valid_voltage_points": 0,
        "delta_q_log_floor_applied": False,
    }
    if cell.q10 is None or cell.q100 is None:
        quality["delta_q_status"] = "Q10 또는 Q100 누락"
        return values, quality
    same_shape = (
        cell.voltage.ndim == 1
        and cell.voltage.shape == cell.q10.shape == cell.q100.shape
        and cell.voltage.size >= 2
    )
    monotonic_voltage = same_shape and (
        np.all(np.diff(cell.voltage) > 0) or np.all(np.diff(cell.voltage) < 0)
    )
    if not same_shape or not np.isfinite(cell.voltage).all() or not monotonic_voltage:
        quality["delta_q_status"] = "전압격자 또는 배열 길이 불일치"
        return values, quality

    valid = np.isfinite(cell.q10) & np.isfinite(cell.q100)
    delta_q = cell.q100[valid] - cell.q10[valid]
    quality["valid_voltage_points"] = int(delta_q.size)
    if delta_q.size < 2:
        quality["delta_q_status"] = "유효 전압 점 부족"
        return values, quality

    variance = float(np.var(delta_q, ddof=0))
    values["delta_q_log_variance"] = float(np.log10(max(variance, 1e-12)))
    values["delta_q_min"] = float(np.min(delta_q))
    quality["delta_q_log_floor_applied"] = variance < 1e-12

    if np.isfinite(mean_qd_early) and mean_qd_early > 0:
        normalized = delta_q / mean_qd_early
        normalized_variance = float(np.var(normalized, ddof=0))
        values["normalized_delta_q_log_variance"] = float(
            np.log10(max(normalized_variance, 1e-12))
        )
        values["normalized_delta_q_min"] = float(np.min(normalized))
    return values, quality


def extract_cell_features(cell: CellRecord) -> tuple[dict, dict]:
    summary = cell.summary.sort_values("cycle").reset_index(drop=True)
    if summary["cycle"].duplicated().any() or not summary["cycle"].between(1, 100).all():
        raise ValueError(f"{cell.cell_id}: 초기 사이클 번호를 확인하세요.")

    capacity = summary.loc[summary["cycle"].between(2, 100), ["cycle", "qd"]]
    capacity = capacity.loc[np.isfinite(capacity["qd"])]
    mean_qd = _valid_mean(capacity["qd"])
    qd_slope = (
        float(np.polyfit(capacity["cycle"], capacity["qd"], 1)[0])
        if len(capacity) >= 2
        else np.nan
    )
    early_qd = _window_mean(summary, "qd", 2, 11)
    late_qd = _window_mean(summary, "qd", 91, 100)
    relative_capacity_change = (
        float((late_qd - early_qd) / early_qd)
        if np.isfinite(early_qd) and early_qd > 0 and np.isfinite(late_qd)
        else np.nan
    )

    valid_temperature_span = (
        summary["temperature_max"].notna() & summary["temperature_min"].notna()
    )
    temperature_span = (
        summary.loc[valid_temperature_span, "temperature_max"]
        - summary.loc[valid_temperature_span, "temperature_min"]
    )
    delta_values, delta_quality = _delta_q_features(cell, early_qd)
    row = {
        "cell_id": cell.cell_id,
        "batch": cell.batch,
        "cycle_life": cell.cycle_life,
        "charging_policy": cell.charging_policy,
        **cell.policy_values,
        **delta_values,
        "mean_qd": mean_qd,
        "qd_slope": qd_slope,
        "relative_capacity_change": relative_capacity_change,
        "mean_ir": _valid_mean(summary["ir"]),
        "ir_change": _difference_between_windows(summary, "ir"),
        "mean_temperature": _valid_mean(summary["temperature_avg"]),
        "mean_temperature_range": _valid_mean(temperature_span),
        "temperature_change": _difference_between_windows(
            summary, "temperature_avg"
        ),
        "mean_charge_time": _valid_mean(summary["charge_time"]),
    }
    quality = {
        "cell_id": cell.cell_id,
        "valid_capacity_cycles": int(len(capacity)),
        "valid_ir_cycles": int(summary["ir"].notna().sum()),
        "valid_temperature_cycles": int(summary["temperature_avg"].notna().sum()),
        **delta_quality,
    }
    return row, quality


def build_feature_table(cells: list[CellRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    quality_rows: list[dict] = []
    for cell in cells:
        row, quality = extract_cell_features(cell)
        rows.append(row)
        quality_rows.append(quality)
    feature_table = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    if feature_table["cell_id"].duplicated().any():
        raise ValueError("피처 표의 셀 식별자가 중복되었습니다.")
    return feature_table, pd.DataFrame(quality_rows)


def initial_feature_sets() -> dict[str, FeatureSet]:
    a = ("delta_q_log_variance",)
    b = (*a, "delta_q_min")
    sets = [
        FeatureSet("A", a, "design", "existing"),
        FeatureSet("B", b, "design", "existing"),
        FeatureSet("C_time", (*b, "mean_charge_time"), "design", "existing"),
        FeatureSet(
            "C_capacity", (*b, "mean_qd", "qd_slope"), "design", "existing"
        ),
        FeatureSet("C_resistance", (*b, "mean_ir"), "design", "existing"),
        FeatureSet(
            "C_temperature",
            (*b, "mean_temperature", "mean_temperature_range"),
            "design",
            "existing",
        ),
        FeatureSet(
            "C_policy",
            (*b, "c_rate_stage1", "c_rate_stage2", "switch_soc_pct"),
            "design",
            "existing",
        ),
        FeatureSet(
            "legacy_capacity_policy",
            ("mean_qd", "qd_slope", "c_rate_stage1", "c_rate_stage2", "switch_soc_pct"),
            "legacy",
            "existing",
        ),
        FeatureSet(
            "legacy_capacity_temperature",
            ("mean_qd", "qd_slope", "mean_temperature", "mean_temperature_range"),
            "legacy",
            "existing",
        ),
        FeatureSet(
            "legacy_capacity_ir",
            ("mean_qd", "qd_slope", "mean_ir"),
            "legacy",
            "existing",
        ),
        FeatureSet("B_without_mean_qd", (*b, "qd_slope"), "changed", "changed"),
        FeatureSet(
            "B_relative_capacity",
            (*b, "relative_capacity_change"),
            "changed",
            "changed",
        ),
        FeatureSet(
            "normalized_delta_q",
            ("normalized_delta_q_log_variance", "normalized_delta_q_min"),
            "changed",
            "changed",
        ),
        FeatureSet(
            "A_temperature_change", (*a, "temperature_change"), "changed", "changed"
        ),
        FeatureSet("A_ir_change", (*a, "ir_change"), "changed", "changed"),
    ]
    return {feature_set.name: feature_set for feature_set in sets}


def expand_feature_set(feature_set: FeatureSet) -> dict[str, FeatureSet]:
    """생존 기준 구성에서 변화 피처·교체·간소화 후보를 만듭니다."""
    candidates: list[tuple[str, tuple[str, ...]]] = []
    columns = tuple(feature_set.columns)
    for suffix, added in (
        ("plus_relative_capacity", "relative_capacity_change"),
        ("plus_ir_change", "ir_change"),
        ("plus_temperature_change", "temperature_change"),
    ):
        if added not in columns:
            candidates.append((f"{feature_set.name}__{suffix}", (*columns, added)))
    if "mean_qd" in columns:
        candidates.append(
            (
                f"{feature_set.name}__drop_mean_qd",
                tuple(column for column in columns if column != "mean_qd"),
            )
        )
    if "qd_slope" in columns:
        candidates.append(
            (
                f"{feature_set.name}__slope_to_relative",
                tuple(
                    "relative_capacity_change" if column == "qd_slope" else column
                    for column in columns
                ),
            )
        )
    if "delta_q_log_variance" in columns:
        normalized = tuple(
            "normalized_delta_q_log_variance"
            if column == "delta_q_log_variance"
            else "normalized_delta_q_min"
            if column == "delta_q_min"
            else column
            for column in columns
        )
        candidates.append((f"{feature_set.name}__normalized_delta", normalized))

    expanded: dict[str, FeatureSet] = {}
    for name, candidate_columns in candidates:
        unique_columns = tuple(dict.fromkeys(candidate_columns))
        if not unique_columns or unique_columns == columns:
            continue
        expanded[name] = FeatureSet(
            name=name,
            columns=unique_columns,
            family=feature_set.family,
            variant="changed",
            parent=feature_set.name,
        )
    return expanded


def all_feature_columns(feature_sets: dict[str, FeatureSet]) -> tuple[str, ...]:
    base_columns = {
        column for feature_set in feature_sets.values() for column in feature_set.columns
    }
    base_columns.update(
        {
            "relative_capacity_change",
            "ir_change",
            "temperature_change",
            "normalized_delta_q_log_variance",
            "normalized_delta_q_min",
        }
    )
    forbidden = {
        "cell_id",
        "batch",
        "cycle_life",
        "charging_policy",
        "protocol_variant",
        "rest_group",
        "cycle5_longest_zero_current_minutes",
        "full_qd",
        "last_cycle",
        "knee",
    }
    if forbidden & base_columns:
        raise ValueError("입력 피처에 식별자·정답·설명용 또는 후기 정보가 포함되었습니다.")
    return tuple(sorted(base_columns))
