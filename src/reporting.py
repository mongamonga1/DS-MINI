"""공통 실험 결과를 터미널과 노트북에 표시합니다."""

from __future__ import annotations

from pathlib import Path

from config import PROJECT_ROOT
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np

from .eda import build_eda_findings
from .experiment import ExperimentResult


MAX_DISPLAY_CANDIDATES = 10


EDA_ASSET_NAMES = (
    "eda_01_life_distribution", "eda_02_initial_signals",
    "eda_03_correlation_rest", "eda_04_full_qd",
    "eda_05_normalized_qd", "eda_06_delta_q_curves",
    "eda_07_delta_q_groups", "eda_08_log_transforms",
    "eda_09_feature_correlation", "eda_10_batch_distributions",
    "eda_11_protocol_life",
)
MODEL_ASSET_NAMES = ("model_01_performance", "model_02_batch2_predictions")


def _save_graph_assets(figures, names):
    """화면에 표시할 그래프를 같은 이름의 PNG로 저장합니다."""
    if len(figures) != len(names):
        raise ValueError("그래프와 저장 파일 이름 개수가 다릅니다.")
    asset_dir = PROJECT_ROOT / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for figure, name in zip(figures, names, strict=True):
        figure.savefig(asset_dir / f"{name}.png", dpi=180, bbox_inches="tight")


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
                "항목": "cycle5 전류·시간 확보 셀",
                "값": int(data["cycle5_current_available"].sum()),
            },
            {
                "항목": "ΔQ 계산 이상 셀",
                "값": int(result.feature_quality["delta_q_status"].ne("정상").sum()),
            },
        ]
    )


def eda_findings(result: ExperimentResult) -> pd.DataFrame:
    return build_eda_findings(
        result.eda_tables,
        result.stage_results,
        result.final_configurations,
        result.error_analysis,
    )


def train_correlation_preview(result: ExperimentResult, n: int = 10) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    return result.eda_tables["train_feature_correlations"].head(limit)


def redundancy_preview(result: ExperimentResult, n: int = 10) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    return result.eda_tables["train_redundant_pairs"].head(limit)


def protocol_comparison_preview(
    result: ExperimentResult, n: int = 10
) -> pd.DataFrame:
    limit = min(max(int(n), 1), MAX_DISPLAY_CANDIDATES)
    table = result.eda_tables["protocol_summary"]
    target = table.loc[
        table["charging_policy"].str.contains(
            "4.8C(80%)-4.8C", regex=False, na=False
        )
    ]
    if target.empty:
        target = table.sort_values(["n", "mean_life"], ascending=[False, True])
    return target.head(limit)


def rest_summary_preview(result: ExperimentResult) -> pd.DataFrame:
    return result.eda_tables["rest_summary"].loc[
        result.eda_tables["rest_summary"]["batch"].eq("Batch 2")
    ]


def plot_eda(result: ExperimentResult) -> list[plt.Figure]:
    korean_font = _configure_font()
    figures: list[plt.Figure] = []
    if not result.eda_tables:
        return figures

    life = result.eda_tables["life_cells"].dropna(subset=["cycle_life"])
    role_labels = {"train": "Train", "valid": "Valid", "test": "Batch2"}
    figure1, axes1 = plt.subplots(figsize=(9, 5))
    for role, label in role_labels.items():
        values = life.loc[life["role"].eq(role), "cycle_life"]
        settings = result.eda_tables["eda_settings"].iloc[0]
        axes1.hist(values, bins=np.linspace(settings.hist_min, settings.hist_max, 25), alpha=0.45, label=label)
    axes1.axvline(500, linestyle="--", color="black", linewidth=1)
    axes1.set_xlabel("총수명(사이클)" if korean_font else "Cycle life")
    axes1.set_ylabel("셀 수" if korean_font else "Cell count")
    axes1.set_title(
        "학습 전 Train·Valid·Batch2 수명 분포"
        if korean_font
        else "Cycle-life distribution before modeling"
    )
    axes1.legend()
    figure1.tight_layout()
    figures.append(figure1)

    train = result.eda_tables["train_feature_values"]
    figure2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    axes2[0].scatter(train["qd_slope"], train["cycle_life"], alpha=0.8)
    axes2[0].axvline(0, linestyle="--", color="gray", linewidth=1)
    axes2[0].set(
        xlabel="Qd slope (Ah/cycle)",
        ylabel="Cycle life",
        title="Train: initial Qd slope",
    )
    axes2[1].scatter(
        train["delta_q_log_variance"], train["cycle_life"], alpha=0.8
    )
    axes2[1].set(
        xlabel="log10 variance of ΔQ",
        ylabel="Cycle life",
        title="Train: ΔQ signal",
    )
    figure2.tight_layout()
    figures.append(figure2)

    correlations = train_correlation_preview(result)
    rest = result.eda_tables["rest_cells"].loc[
        result.eda_tables["rest_cells"]["batch"].eq("Batch 2")
    ].dropna(
        subset=["cycle_life", "cycle5_longest_zero_current_minutes"]
    )
    figure3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    if not correlations.empty:
        axes3[0].barh(
            correlations["feature"][::-1], correlations["spearman"][::-1]
        )
    axes3[0].set(
        xlabel="Spearman rho",
        title="Train-only feature correlation",
        xlim=(-1, 1),
    )
    for batch, group in rest.groupby("batch", sort=False):
        axes3[1].scatter(
            group["cycle5_longest_zero_current_minutes"],
            group["cycle_life"],
            alpha=0.75,
            label=batch,
        )
    rest_threshold = result.eda_tables["rest_config"].iloc[0][
        "long_rest_threshold_minutes"
    ]
    axes3[1].axvline(
        rest_threshold, linestyle="--", color="black", linewidth=1
    )
    axes3[1].set(
        xlabel=f"Cycle5 longest |I|<{result.eda_tables['rest_config'].iloc[0]['current_threshold_amp']:g}A interval (min)",
        ylabel="Cycle life",
        title="Rest-pattern diagnostic only",
    )
    axes3[1].legend()
    figure3.tight_layout()
    figures.append(figure3)

    trajectories = result.eda_tables["full_qd_trajectories"].dropna(subset=["qd"])
    figure4, axes4 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axes, batch in zip(axes4, ("Batch 1", "Batch 2"), strict=True):
        batch_values = trajectories.loc[trajectories["batch"].eq(batch)]
        for _, group in batch_values.groupby("cell_id", sort=False):
            axes.plot(group["cycle"], group["qd"], alpha=0.12, linewidth=0.8)
        axes.set_title(f"{batch}: full Qd (EDA only)")
        axes.set_xlabel("Cycle")
    axes4[0].set_ylabel("Qd (Ah)")
    figure4.tight_layout()
    figures.append(figure4)
    figures.extend(_additional_eda_plots(result))
    _save_graph_assets(figures, EDA_ASSET_NAMES)
    return figures


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
    _save_graph_assets(figures, MODEL_ASSET_NAMES[:len(figures)])
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
    print("\n[EDA → 후보 → 검증 연결]")
    print(eda_findings(result).to_string(index=False))
    print("\n[학습 전 수명 분포]")
    print(result.eda_tables["life_distribution"].to_string(index=False))
    print("\n[Train 초기 용량 변화]")
    print(result.eda_tables["capacity_behavior"].to_string(index=False))
    print("\n[Train 피처-수명 상관(최대 10개)]")
    print(train_correlation_preview(result).to_string(index=False))
    print("\n[Train 중복 피처쌍(최대 10개)]")
    print(redundancy_preview(result).to_string(index=False))
    print("\n[4.8C 프로토콜 원문 비교]")
    print(protocol_comparison_preview(result).to_string(index=False))
    print("\n[cycle5 무전류 그룹]")
    print(rest_summary_preview(result).to_string(index=False))
    for name in ["policy_correlations", "batch_feature_coverage", "protocol_coverage"]:
        print(f"\n[EDA: {name}]")
        print(result.eda_tables[name].to_string(index=False))
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
    if not result.model_performance.empty:
        print("\n[프로토콜·무전류·수명 그룹별 오차]")
        print(result.error_analysis["group_metrics"].to_string(index=False))
        print("\n[Train 피처 범위 밖 Batch2 셀]")
        print(feature_range_preview(result).to_string(index=False))
    if show_plots:
        if bool(result.eda_tables["eda_settings"].iloc[0]["show_plots"]):
            plot_eda(result)
        plot_results(result)
        plt.show()


def _additional_eda_plots(result: ExperimentResult) -> list[plt.Figure]:
    tables = result.eda_tables
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    trajectories = tables["full_qd_trajectories"]
    for ax, batch in zip(axes, ("Batch 1", "Batch 2")):
        for _, g in trajectories.loc[trajectories.batch.eq(batch)].groupby("cell_id"):
            early = g.loc[g.cycle.between(2, 11), "qd"].dropna()
            base = early.mean()
            if np.isfinite(base) and base > 0:
                g = g.loc[g.cycle.ge(2)]
                ax.plot(g.cycle, g.qd / base, alpha=.15, linewidth=.8)
        ax.set(title=f"{batch}: Qd / own cycles 2-11 mean (EDA only)", xlabel="Cycle", ylabel="Relative capacity")
    fig.tight_layout(); figures.append(fig)

    curves = tables["delta_q_curves"]
    groups = ["<500", "500-1000", ">1000"]
    colors = ["tab:red", "tab:gray", "tab:blue"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, batch in zip(axes, ("Batch 1", "Batch 2")):
        for group, color in zip(groups, colors):
            sub = curves.loc[curves.batch.eq(batch) & curves.life_group.eq(group)]
            for _, g in sub.groupby("cell_id"):
                ax.plot(g.voltage, g.delta_q, color=color, alpha=.13, linewidth=.7)
            if not sub.empty:
                median = sub.groupby("voltage").delta_q.median()
                ax.plot(median.index, median, color=color, label=f"{group}, n={sub.cell_id.nunique()}")
        ax.set(title=f"{batch}: Q100-Q10 (descriptive)", xlabel="Voltage (V)", ylabel="ΔQ (Ah)")
        ax.legend()
    fig.tight_layout(); figures.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    delta = tables["delta_q_groups"]
    for ax, column in zip(axes, ["delta_q_log_variance", "delta_q_min"]):
        values, labels = [], []
        for (batch, group), g in delta.groupby(["batch", "life_group"]):
            v = g[column].dropna()
            if len(v): values.append(v); labels.append(f"{batch} {group}\nn={len(v)}")
        if values: ax.boxplot(values, tick_labels=labels)
        ax.set_title(column); ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout(); figures.append(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    train = tables["train_transform_values"]
    for ax, column in zip(axes.flat, ["cycle_life", "log10_cycle_life", "delta_q_variance", "delta_q_log_variance"]):
        ax.hist(train[column].dropna(), bins=12)
        ax.set(title=f"Train only: {column}", ylabel="Cell count")
    fig.tight_layout(); figures.append(fig)

    matrix = tables["train_feature_correlation_matrix"]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_title("Train-only feature Spearman correlation")
    fig.colorbar(im, ax=ax); fig.tight_layout(); figures.append(fig)

    values = tables["batch_feature_distribution"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    columns = ["delta_q_log_variance", "delta_q_min", "mean_qd", "mean_ir", "mean_temperature", "mean_charge_time"]
    for ax, column in zip(axes.flat, columns):
        arrays = [values.loc[values.batch.eq(b), column].dropna() for b in ["Batch 1", "Batch 2"]]
        ax.boxplot(arrays, tick_labels=["Batch 1", "Batch 2"])
        ax.set_title(f"{column} (descriptive)")
    fig.tight_layout(); figures.append(fig)

    policies = tables["protocol_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, batch in zip(axes, ["Batch 1", "Batch 2"]):
        g = policies.loc[policies.batch.eq(batch)].sort_values("mean_life")
        ax.barh(range(len(g)), g.mean_life)
        ax.set_yticks(range(len(g)), [f"{r.charging_policy} (n={r.n})" for r in g.itertuples()], fontsize=7)
        ax.set(title=f"{batch}: protocol mean life", xlabel="Cycle life")
    fig.tight_layout(); figures.append(fig)
    return figures
