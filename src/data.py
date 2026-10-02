"""원본 확보, 읽기, 품질 처리, 셀 단위 분할."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import urllib.request

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from config import ExperimentConfig


BATCH_NAMES = {
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat": "Batch 1",
    "2018-02-20_batchdata_updated_struct_errorcorrect.mat": "Batch 2",
}
SUMMARY_FIELDS = {
    "qd": "QDischarge",
    "ir": "IR",
    "temperature_avg": "Tavg",
    "temperature_min": "Tmin",
    "temperature_max": "Tmax",
    "charge_time": "chargetime",
}


@dataclass
class CellRecord:
    cell_id: str
    batch: str
    cycle_life: float
    charging_policy: str
    barcode: str
    policy_values: dict[str, float]
    summary: pd.DataFrame
    voltage: np.ndarray
    q10: np.ndarray | None
    q100: np.ndarray | None
    full_qd: pd.DataFrame
    cycle5_time: np.ndarray | None
    cycle5_current: np.ndarray | None


@dataclass
class DataBundle:
    cells: list[CellRecord]
    metadata: pd.DataFrame
    quality: pd.DataFrame
    data_dir: Path


@dataclass
class SplitBundle:
    train_ids: tuple[str, ...]
    valid_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    unlabeled_ids: tuple[str, ...]
    assignments: pd.DataFrame
    cv_folds: pd.DataFrame


def _has_required_files(directory: Path, filenames: tuple[str, ...]) -> bool:
    return all((directory / filename).is_file() for filename in filenames)


def prepare_data(config: ExperimentConfig) -> Path:
    """원본 위치를 확인하고, 검증된 주소가 있을 때만 내려받습니다."""
    configured = config.data_dir.resolve()
    if _has_required_files(configured, config.batch_files):
        return configured

    local_candidates = (
        config.project_root.parent / "DS-MINI-Day2" / "data" / "raw",
        config.project_root.parent / "0-Data-scratch" / "data",
    )
    for candidate in local_candidates:
        if _has_required_files(candidate, config.batch_files):
            return candidate.resolve()

    configured.mkdir(parents=True, exist_ok=True)
    missing = [name for name in config.batch_files if not (configured / name).is_file()]
    for filename in list(missing):
        url = config.download_urls.get(filename)
        if not url:
            continue
        temporary = configured / f".{filename}.download"
        try:
            urllib.request.urlretrieve(url, temporary)
            shutil.move(temporary, configured / filename)
            missing.remove(filename)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"원본 다운로드에 실패했습니다: {filename}")

    if missing:
        expected = "\n".join(f"- {filename}" for filename in missing)
        raise FileNotFoundError(
            "필요한 원본 파일이 없습니다. data/raw/에 다음 파일을 배치하거나 "
            "ESS_DATA_DIR을 설정하세요.\n"
            f"{expected}\n"
            "검증된 직접 다운로드 주소가 설정되지 않아 다른 배치로 대체하지 않습니다."
        )
    return configured


def _reference_item(raw_file: h5py.File, group: h5py.Group, name: str, index: int):
    references = group[name][()].reshape(-1)
    return raw_file[references[index]]


def _numeric_vector(raw_file: h5py.File, dataset: h5py.Dataset) -> np.ndarray:
    if h5py.check_dtype(ref=dataset.dtype):
        references = dataset[()].reshape(-1)
        if references.size != 1:
            raise ValueError("단일 숫자 배열 참조를 예상했습니다.")
        dataset = raw_file[references[0]]
    return np.asarray(dataset[()], dtype=float).reshape(-1)


def _read_text(raw_file: h5py.File, dataset: h5py.Dataset) -> str:
    if h5py.check_dtype(ref=dataset.dtype):
        references = dataset[()].reshape(-1)
        if references.size != 1:
            return ""
        dataset = raw_file[references[0]]
    values = np.asarray(dataset[()]).reshape(-1)
    if dataset.attrs.get("MATLAB_class") not in (b"char", "char"):
        return ""
    return "".join(chr(int(value)) for value in values if value > 0).strip()


def _parse_policy(text: str) -> tuple[dict[str, float], str]:
    number = r"([0-9]+(?:\.[0-9]+)?)"
    matched = re.match(number + r"C\(" + number + r"%\)-" + number + r"C", text)
    if matched is None:
        return {
            "c_rate_stage1": np.nan,
            "c_rate_stage2": np.nan,
            "switch_soc_pct": np.nan,
        }, "충전 조건 해석 실패"
    return {
        "c_rate_stage1": float(matched.group(1)),
        "switch_soc_pct": float(matched.group(2)),
        "c_rate_stage2": float(matched.group(3)),
    }, "정상"


def load_battery_data(config: ExperimentConfig) -> DataBundle:
    """Batch1·2의 초기 100사이클과 Q10·Q100만 읽습니다."""
    data_dir = prepare_data(config)
    cells: list[CellRecord] = []
    quality_rows: list[dict[str, object]] = []

    for filename in config.batch_files:
        batch_name = BATCH_NAMES.get(filename)
        if batch_name is None:
            raise ValueError(f"허용되지 않은 배치 파일입니다: {filename}")
        path = data_dir / filename
        with h5py.File(path, "r") as raw_file:
            batch = raw_file["batch"]
            for cell_number in range(batch["cycle_life"].size):
                cell_id = f"{path.stem}:cell{cell_number:03d}"
                raw_summary = _reference_item(raw_file, batch, "summary", cell_number)
                raw_cycles = _reference_item(raw_file, batch, "cycles", cell_number)
                cycles = _numeric_vector(raw_file, raw_summary["cycle"])
                if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
                    raise ValueError(f"{cell_id}: 사이클 번호와 곡선 대응을 확인하세요.")
                if raw_cycles["Qdlin"].size != len(cycles):
                    raise ValueError(f"{cell_id}: 요약 행과 곡선 수가 다릅니다.")

                early_mask = (cycles >= 1) & (cycles <= 100)
                summary = pd.DataFrame({"cycle": cycles[early_mask]})
                qd_values = np.full(cycles.shape, np.nan, dtype=float)
                for analysis_name, raw_name in SUMMARY_FIELDS.items():
                    values = _numeric_vector(raw_file, raw_summary[raw_name])
                    if len(values) != len(cycles):
                        raise ValueError(f"{cell_id}: {raw_name} 길이가 다릅니다.")
                    if analysis_name == "qd":
                        qd_values = values
                    summary[analysis_name] = values[early_mask]
                full_qd = pd.DataFrame({"cycle": cycles, "qd": qd_values})

                first_cycle_empty = batch_name == "Batch 1" and bool(
                    (summary["cycle"].eq(1) & summary[
                        ["qd", "temperature_avg", "temperature_min", "temperature_max", "charge_time"]
                    ].eq(0).all(axis=1)).any()
                )
                if first_cycle_empty:
                    first = summary["cycle"].eq(1)
                    summary.loc[
                        first,
                        ["temperature_avg", "temperature_min", "temperature_max", "charge_time"],
                    ] = np.nan
                zero_ir_count = int(summary["ir"].eq(0).sum())
                summary.loc[summary["ir"].eq(0), "ir"] = np.nan

                damaged_temperature_rows = 0
                if cell_id in config.corrupted_temperature_cells:
                    damaged = summary["temperature_min"].eq(-270) | summary[
                        "temperature_max"
                    ].eq(400)
                    damaged_temperature_rows = int(damaged.sum())
                    summary.loc[
                        damaged,
                        ["temperature_avg", "temperature_min", "temperature_max"],
                    ] = np.nan

                voltage = _numeric_vector(
                    raw_file, _reference_item(raw_file, batch, "Vdlin", cell_number)
                )
                curves: dict[int, np.ndarray] = {}
                for cycle_number in (10, 100):
                    positions = np.flatnonzero(cycles == cycle_number)
                    if positions.size == 1:
                        curves[cycle_number] = _numeric_vector(
                            raw_file,
                            _reference_item(
                                raw_file, raw_cycles, "Qdlin", int(positions[0])
                            ),
                        )

                cycle5_time = None
                cycle5_current = None
                cycle5_positions = np.flatnonzero(cycles == 5)
                if (
                    cycle5_positions.size == 1
                    and "I" in raw_cycles
                    and "t" in raw_cycles
                ):
                    cycle_index = int(cycle5_positions[0])
                    current_values = _numeric_vector(
                        raw_file,
                        _reference_item(raw_file, raw_cycles, "I", cycle_index),
                    )
                    time_values = _numeric_vector(
                        raw_file,
                        _reference_item(raw_file, raw_cycles, "t", cycle_index),
                    )
                    if len(current_values) == len(time_values):
                        cycle5_current = current_values
                        cycle5_time = time_values

                life_values = _numeric_vector(
                    raw_file, _reference_item(raw_file, batch, "cycle_life", cell_number)
                )
                if life_values.size != 1:
                    raise ValueError(f"{cell_id}: 수명 라벨의 모양이 다릅니다.")
                raw_life = float(life_values[0])
                cycle_life = raw_life if np.isfinite(raw_life) and raw_life > 0 else np.nan
                policy_text = _read_text(
                    raw_file,
                    _reference_item(raw_file, batch, "policy_readable", cell_number),
                ) or "unknown"
                policy_values, policy_status = _parse_policy(policy_text)
                barcode = (
                    _read_text(
                        raw_file,
                        _reference_item(raw_file, batch, "barcode", cell_number),
                    )
                    if "barcode" in batch
                    else ""
                )

                cells.append(
                    CellRecord(
                        cell_id=cell_id,
                        batch=batch_name,
                        cycle_life=cycle_life,
                        charging_policy=policy_text,
                        barcode=barcode,
                        policy_values=policy_values,
                        summary=summary,
                        voltage=voltage,
                        q10=curves.get(10),
                        q100=curves.get(100),
                        full_qd=full_qd,
                        cycle5_time=cycle5_time,
                        cycle5_current=cycle5_current,
                    )
                )
                quality_rows.append(
                    {
                        "cell_id": cell_id,
                        "batch": batch_name,
                        "label_available": np.isfinite(cycle_life),
                        "first_cycle_empty_values_to_nan": first_cycle_empty,
                        "zero_ir_values_to_nan": zero_ir_count,
                        "damaged_temperature_rows_to_nan": damaged_temperature_rows,
                        "policy_status": policy_status,
                        "barcode_status": "확인 가능" if barcode else "원본 문자열 해석 불가",
                        "q10_available": 10 in curves,
                        "q100_available": 100 in curves,
                        "cycle5_current_available": cycle5_current is not None,
                    }
                )

    metadata = pd.DataFrame(
        [
            {
                "cell_id": cell.cell_id,
                "batch": cell.batch,
                "cycle_life": cell.cycle_life,
                "charging_policy": cell.charging_policy,
                "barcode": cell.barcode,
            }
            for cell in cells
        ]
    )
    if metadata["cell_id"].duplicated().any():
        raise ValueError("셀 식별자가 중복되었습니다.")
    decoded_barcodes = metadata.loc[metadata["barcode"].ne(""), "barcode"]
    if decoded_barcodes.duplicated().any():
        raise ValueError("동일한 원본 바코드를 가진 셀이 중복되었습니다.")
    if metadata.groupby("batch").size().to_dict() != {"Batch 1": 46, "Batch 2": 47}:
        raise ValueError("예상한 Batch1·2 셀 수와 다릅니다.")
    return DataBundle(cells, metadata, pd.DataFrame(quality_rows), data_dir)


def make_cell_splits(metadata: pd.DataFrame, config: ExperimentConfig) -> SplitBundle:
    """셀 목록을 한 번 분할하고 모든 후보가 같은 CV fold를 사용하게 합니다."""
    batch1_ids = sorted(
        metadata.loc[
            metadata["batch"].eq("Batch 1") & metadata["cycle_life"].notna(),
            "cell_id",
        ].tolist()
    )
    train_ids, valid_ids = train_test_split(
        batch1_ids,
        test_size=config.valid_size,
        random_state=config.random_state,
        shuffle=True,
    )
    train_ids = tuple(sorted(train_ids))
    valid_ids = tuple(sorted(valid_ids))
    test_ids = tuple(
        sorted(
            metadata.loc[
                metadata["batch"].eq("Batch 2") & metadata["cycle_life"].notna(),
                "cell_id",
            ].tolist()
        )
    )
    unlabeled_ids = tuple(sorted(metadata.loc[metadata["cycle_life"].isna(), "cell_id"]))

    id_sets = [set(train_ids), set(valid_ids), set(test_ids), set(unlabeled_ids)]
    for left_index, left_ids in enumerate(id_sets):
        for right_ids in id_sets[left_index + 1 :]:
            if left_ids & right_ids:
                raise ValueError("셀 분할 사이에 중복 식별자가 있습니다.")
    if (len(train_ids), len(valid_ids), len(test_ids)) != (36, 10, 39):
        raise ValueError("예상한 36/10/39 셀 분할과 다릅니다.")

    assignment_rows = [
        *({"cell_id": cell_id, "role": "train"} for cell_id in train_ids),
        *({"cell_id": cell_id, "role": "valid"} for cell_id in valid_ids),
        *({"cell_id": cell_id, "role": "test"} for cell_id in test_ids),
        *({"cell_id": cell_id, "role": "unlabeled"} for cell_id in unlabeled_ids),
    ]
    assignments = pd.DataFrame(assignment_rows).merge(
        metadata[["cell_id", "batch", "cycle_life"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    fold_rows: list[dict[str, object]] = []
    cross_validator = KFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    train_array = np.asarray(train_ids)
    for fold_number, (_, validation_indices) in enumerate(
        cross_validator.split(train_array), start=1
    ):
        for cell_id in train_array[validation_indices]:
            fold_rows.append({"cell_id": cell_id, "validation_fold": fold_number})
    cv_folds = pd.DataFrame(fold_rows)
    if not cv_folds["cell_id"].is_unique or set(cv_folds["cell_id"]) != set(train_ids):
        raise ValueError("CV fold 생성 결과를 확인하세요.")
    return SplitBundle(
        train_ids, valid_ids, test_ids, unlabeled_ids, assignments, cv_folds
    )
