"""공통 실험 결과를 터미널과 노트북에 표시합니다."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

from .experiment import ExperimentResult


MAX_DISPLAY_CANDIDATES = 10


def _configure_font() -> bool:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("AppleGothic", "Malgun Gothic", "NanumGothic"):
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def stage_summary(result: ExperimentResult) -> pd.DataFrame:
    rows = []
    for stage_name, table in result.stage_results.items():
        selection = result.selection_history[stage_name]
        rows.append(
            {
                "단계": stage_name,
                "평가 후보": len(table),
                "정상 완료": int(table["status"].eq("ok").sum()),
                "경고·실패": int((~table["status"].eq("ok")).sum()),
                "생존 후보": int(selection["selected"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _representative_rows(table: pd.DataFrame, n: int) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    ranked = table.sort_values(
        ["cv_mape_mean", "cv_mape_std", "n_features", "candidate_id"],
        kind="stable",
        na_position="last",
    )
    if "status" in ranked.columns:
        ranked = ranked.loc[ranked["status"].eq("ok")]
    if ranked.empty:
        return ranked.head(0)

    diversity_columns = [
        column
        for column in ["model", "target_transform", "family", "variant"]
        if column in ranked.columns
    ]
    seen = {column: set() for column in diversity_columns}
    selected_indices = []
    for row_index, row in ranked.iterrows():
        introduces_characteristic = not selected_indices or any(
            str(row[column]) not in seen[column] for column in diversity_columns
        )
        if not introduces_characteristic:
            continue
        selected_indices.append(row_index)
        for column in diversity_columns:
            seen[column].add(str(row[column]))
        if len(selected_indices) == limit:
            break

    if len(selected_indices) < limit:
        for row_index in ranked.index:
            if row_index in selected_indices:
                continue
            selected_indices.append(row_index)
            if len(selected_indices) == limit:
                break
    return ranked.loc[selected_indices]


def top_candidates(result: ExperimentResult, stage_name: str, n: int = 10) -> pd.DataFrame:
    table = _representative_rows(result.stage_results[stage_name], n)
    columns = [
        "feature_set",
        "family",
        "variant",
        "model",
        "target_transform",
        "params",
        "cv_mape_mean",
        "cv_mape_std",
        "cv_mae_mean",
        "cv_rmse_mean",
        "status",
    ]
    return table.loc[:, columns]


def selected_candidates(
    result: ExperimentResult, stage_name: str, n: int = 10
) -> pd.DataFrame:
    selection = result.selection_history[stage_name]
    selected = selection.loc[selection["selected"]]
    table = _representative_rows(selected, n)
    columns = [
        column
        for column in [
            "feature_set",
            "family",
            "variant",
            "model",
            "target_transform",
            "params",
            "branch_rank",
            "global_rank",
            "cv_mape_mean",
            "cv_mape_std",
            "selection_reason",
        ]
        if column in table.columns
    ]
    return table.loc[:, columns]


def candidate_preview(
    result: ExperimentResult, stage_name: str, n: int = 10
) -> pd.DataFrame:
    table = _representative_rows(result.selection_history[stage_name], n)
    columns = [
        column
        for column in [
            "feature_set",
            "family",
            "variant",
            "model",
            "target_transform",
            "params",
            "branch_rank",
            "global_rank",
            "cv_mape_mean",
            "cv_mape_std",
            "selected",
            "selection_reason",
        ]
        if column in table.columns
    ]
    return table.loc[:, columns]


def feature_catalog_preview(
    result: ExperimentResult, n: int = 10
) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    table = result.feature_catalog.sort_values(
        ["n_features", "family", "variant", "feature_set"], kind="stable"
    )
    representatives = table.drop_duplicates(["family", "variant"], keep="first")
    remaining = table.loc[~table.index.isin(representatives.index)]
    return pd.concat([representatives, remaining], axis=0).head(limit)


def performance_summary(result: ExperimentResult) -> pd.DataFrame:
    if result.model_performance.empty:
        return result.model_performance.copy()
    metrics = ["mape_pct", "mae_cycles", "rmse_cycles"]
    pivot = result.model_performance.pivot(
        index="configuration", columns="dataset", values=metrics
    )
    pivot.columns = [f"{dataset}_{metric}" for metric, dataset in pivot.columns]
    return pivot.reset_index()


def gap_summary(result: ExperimentResult) -> pd.DataFrame:
    if result.gaps.empty:
        return result.gaps.copy()
    return result.gaps.pivot(
        index="configuration", columns="gap", values="value_pct_point"
    ).reset_index()


def group_error_preview(result: ExperimentResult, n: int = 10) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    table = result.error_analysis["group_metrics"].sort_values(
        ["mape_pct", "n", "group"],
        ascending=[False, False, True],
        kind="stable",
    )
    representatives = table.drop_duplicates("group_type", keep="first")
    remaining = table.loc[~table.index.isin(representatives.index)]
    return pd.concat([representatives, remaining], axis=0).head(limit)


def feature_range_preview(result: ExperimentResult, n: int = 10) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    return (
        result.error_analysis["feature_range"]
        .sort_values(
            ["test_outside_count", "feature"],
            ascending=[False, True],
            kind="stable",
            na_position="last",
        )
        .head(limit)
    )


def quality_summary(result: ExperimentResult) -> pd.DataFrame:
    data = result.data_quality
    return pd.DataFrame(
        [
            {"항목": "전체 셀", "값": len(data)},
            {"항목": "수명 라벨 보유", "값": int(data["label_available"].sum())},
            {
                "항목": "Batch1 첫 사이클 빈 측정 처리 셀",
                "값": int(data["first_cycle_empty_values_to_nan"].sum()),
            },
            {
                "항목": "IR 0 제외 건수",
                "값": int(data["zero_ir_values_to_nan"].sum()),
            },
            {
                "항목": "손상 온도 처리 행",
                "값": int(data["damaged_temperature_rows_to_nan"].sum()),
            },
            {
                "항목": "충전 조건 해석 실패 셀",
                "값": int(data["policy_status"].ne("정상").sum()),
            },
            {
                "항목": "바코드 해석 불가 셀",
                "값": int(data["barcode_status"].ne("확인 가능").sum()),
            },
            {
                "항목": "ΔQ 계산 이상 셀",
                "값": int(result.feature_quality["delta_q_status"].ne("정상").sum()),
            },
        ]
    )


def plot_results(result: ExperimentResult) -> list[plt.Figure]:
    korean_font = _configure_font()
    figures: list[plt.Figure] = []
    if result.model_performance.empty:
        return figures

    performance = result.model_performance.copy()
    pivot = performance.pivot(index="configuration", columns="dataset", values="mape_pct")
    figure1, axes1 = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=axes1)
    axes1.set_ylabel("MAPE (%)")
    axes1.set_title(
        "구성별 Train CV·Valid·Batch2 성능"
        if korean_font
        else "Train CV, Valid, and Batch2 by configuration"
    )
    axes1.tick_params(axis="x", rotation=20)
    figure1.tight_layout()
    figures.append(figure1)

    predictions = result.predictions.get("final:Batch2")
    if predictions is not None and not predictions.empty:
        figure2, axes2 = plt.subplots(figsize=(6, 5))
        axes2.scatter(predictions["actual"], predictions["predicted"], alpha=0.8)
        lower = float(predictions[["actual", "predicted"]].min().min())
        upper = float(predictions[["actual", "predicted"]].max().max())
        axes2.plot([lower, upper], [lower, upper], linestyle="--", color="gray")
        axes2.set_xlabel("실제 수명(사이클)" if korean_font else "Actual cycle life")
        axes2.set_ylabel("예측 수명(사이클)" if korean_font else "Predicted cycle life")
        axes2.set_title(
            "최종 구성의 Batch2 예측"
            if korean_font
            else "Final configuration predictions on Batch2"
        )
        figure2.tight_layout()
        figures.append(figure2)
    return figures


def print_terminal_report(result: ExperimentResult, *, show_plots: bool = True) -> None:
    print("\n[환경]")
    print(result.environment.to_string(index=False))
    print("\n[설정]")
    print(result.settings.to_string(index=False))
    print("\n[데이터 품질]")
    print(quality_summary(result).to_string(index=False))
    print("\n[분할]")
    print(result.split_assignments.groupby(["batch", "role"]).size().rename("셀 수"))
    print("\n[단계별 후보]")
    print(stage_summary(result).to_string(index=False))
    for stage_name in result.stage_results:
        print(f"\n[{stage_name} 대표 후보(최대 {MAX_DISPLAY_CANDIDATES}개)]")
        print(candidate_preview(result, stage_name).to_string(index=False))
        selection = result.selection_history[stage_name]
        print(f"\n[{stage_name} 선별 사유 집계]")
        print(selection["selection_reason"].value_counts(dropna=False).to_string())
    if not result.final_configurations.empty:
        print("\n[최종·대표 구성]")
        print(result.final_configurations.to_string(index=False))
        print("\n[성능]")
        print(performance_summary(result).to_string(index=False))
        print("\n[GAP]")
        print(gap_summary(result).to_string(index=False))
        print("\n[500사이클 미만 오류]")
        print(result.error_analysis["short_life_summary"].to_string(index=False))
        print("\n[절대오차 상위 셀]")
        print(
            result.error_analysis["worst_absolute"][[
                "cell_id", "actual", "predicted", "signed_error", "ape_pct"
            ]].to_string(index=False)
        )
    if show_plots:
        plot_results(result)
        plt.show()
