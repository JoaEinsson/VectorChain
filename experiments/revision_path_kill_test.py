"""Short eliminatory test of predictive information in full revision paths."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import revisable_chain
import revisable_chain_validation as validation

GEOMETRY_WIDTH = 17
PATH_COORDINATES = 8
PATH_SIGNATURE_WIDTH = PATH_COORDINATES + PATH_COORDINATES**2
PATH_STATISTICS_WIDTH = PATH_COORDINATES * 3 + 8
PATH_DESCRIPTOR_WIDTH = PATH_SIGNATURE_WIDTH + PATH_STATISTICS_WIDTH
PATH_INPUT_WIDTH = GEOMETRY_WIDTH + PATH_DESCRIPTOR_WIDTH
LAST_UPDATE_WIDTH = GEOMETRY_WIDTH + PATH_COORDINATES
REGISTERED_RAW_LAGS = PATH_INPUT_WIDTH - 1
REPRESENTATIONS = (
    "geometry",
    "geometry_last_update",
    "geometry_revision_path",
    "geometry_sham_path",
    "raw_matched",
)
COMPARATORS = {
    "geometry": "geometry",
    "last_update": "geometry_last_update",
    "sham_path": "geometry_sham_path",
    "raw_matched": "raw_matched",
}


@dataclass(frozen=True, slots=True)
class KillTestConfig:
    """Validated configuration for the one-shot revision-path kill test."""

    name: str
    phase: str
    seeds: tuple[int, ...]
    mechanisms: tuple[str, ...]
    n_points: int
    noise_std: float
    lambda_revision: float
    lambda_bend: float
    train_end: int
    validation_end: int
    test_end: int
    raw_lags: int
    model_kind: str
    alphas: tuple[float, ...]
    path_ratio_max: float
    raw_ratio_max: float
    required_seed_passes: int
    required_mechanism_passes: int
    required_horizon_passes: int
    output_root: str
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RevisionPathDesign:
    """Aligned causal inputs and targets for one generated signal."""

    mechanism: str
    seed: int
    origins: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    targets: NDArray[np.float64]
    inputs: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class KillTestSummary:
    """Paths and decision emitted by a completed eliminatory run."""

    run_id: str
    run_dir: Path
    gate_path: Path
    passed: bool


def load_config(path: Path) -> KillTestConfig:
    """Load and validate every choice used by the eliminatory experiment."""

    with path.resolve().open("rb") as stream:
        raw = tomllib.load(stream)
    experiment = _table(raw, "experiment")
    signals = _table(raw, "signals")
    revision = _table(raw, "revision")
    split = _table(raw, "split")
    representation = _table(raw, "representation")
    model = _table(raw, "model")
    gate = _table(raw, "gate")
    output = _table(raw, "output")

    mechanisms = _string_tuple(signals.get("names"), "signals.names")
    if any(name not in revisable_chain.MECHANISM_NAMES for name in mechanisms):
        raise ValueError("signals.names contains an unregistered K7 mechanism")
    train_end = _integer(split.get("train_end"), "split.train_end", minimum=1)
    validation_end = _integer(split.get("validation_end"), "split.validation_end", minimum=2)
    test_end = _integer(split.get("test_end"), "split.test_end", minimum=3)
    n_points = _integer(signals.get("n_points"), "signals.n_points", minimum=512)
    if not train_end < validation_end < test_end or test_end != n_points:
        raise ValueError("split endpoints must increase and test_end must equal signals.n_points")

    lambda_revision = _real(revision.get("lambda_revision"), "revision.lambda_revision")
    lambda_bend = _real(revision.get("lambda_bend"), "revision.lambda_bend")
    if (lambda_revision, lambda_bend) != (0.1, 1.0):
        raise ValueError("revision penalties must reuse the locked K7 pair (0.1, 1.0)")
    raw_lags = _integer(representation.get("raw_lags"), "representation.raw_lags", minimum=1)
    if raw_lags != REGISTERED_RAW_LAGS:
        raise ValueError(f"representation.raw_lags must equal {REGISTERED_RAW_LAGS}")
    model_kind = _string(model.get("kind"), "model.kind")
    if model_kind != "ridge":
        raise ValueError("model.kind must be 'ridge'")
    alphas = _real_tuple(model.get("alpha"), "model.alpha")
    if any(alpha <= 0.0 for alpha in alphas):
        raise ValueError("model.alpha values must be positive")

    return KillTestConfig(
        name=_string(experiment.get("name"), "experiment.name"),
        phase=_string(experiment.get("phase"), "experiment.phase"),
        seeds=_integer_tuple(experiment.get("seeds"), "experiment.seeds"),
        mechanisms=mechanisms,
        n_points=n_points,
        noise_std=_real(signals.get("noise_std"), "signals.noise_std"),
        lambda_revision=lambda_revision,
        lambda_bend=lambda_bend,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
        raw_lags=raw_lags,
        model_kind=model_kind,
        alphas=alphas,
        path_ratio_max=_real(gate.get("path_ratio_max"), "gate.path_ratio_max"),
        raw_ratio_max=_real(gate.get("raw_ratio_max"), "gate.raw_ratio_max"),
        required_seed_passes=_integer(
            gate.get("required_seed_passes"), "gate.required_seed_passes", minimum=1
        ),
        required_mechanism_passes=_integer(
            gate.get("required_mechanism_passes"),
            "gate.required_mechanism_passes",
            minimum=1,
        ),
        required_horizon_passes=_integer(
            gate.get("required_horizon_passes"), "gate.required_horizon_passes", minimum=1
        ),
        output_root=_string(output.get("root"), "output.root"),
        raw=raw,
    )


def build_revision_path_design(
    bundle: revisable_chain.K7DesignBundle, *, raw_lags: int = REGISTERED_RAW_LAGS
) -> RevisionPathDesign:
    """Add the complete current-link revision lineage to the current geometry."""

    if raw_lags != REGISTERED_RAW_LAGS:
        raise ValueError(f"raw_lags must equal {REGISTERED_RAW_LAGS}")
    versions = {version.observed_at: version for version in bundle.versions}
    histories: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for version in bundle.versions:
        for link in version.links:
            if link.update_theta != 0.0 or link.update_r != 0.0:
                histories[link.link_id].append(
                    (version.observed_at, link.update_theta, link.update_r)
                )

    descriptors: list[NDArray[np.float64]] = []
    latest_updates: list[NDArray[np.float64]] = []
    keep_rows: list[int] = []
    raw_inputs: list[NDArray[np.float64]] = []
    values = bundle.signal.values
    for row_index, origin_value in enumerate(bundle.origins):
        origin = int(origin_value)
        if origin < raw_lags:
            continue
        version = versions[origin]
        if len(version.links) != 4:
            raise RuntimeError("revision-path origins require exactly four current links")
        increments, counts = _current_link_revision_increments(version, histories)
        ages = np.asarray(
            [origin - link.created_at + 1 for link in version.links], dtype=np.float64
        )
        descriptors.append(_revision_path_descriptor(increments, ages=ages, counts=counts))
        latest_updates.append(
            np.asarray(
                [value for link in version.links for value in (link.update_theta, link.update_r)],
                dtype=np.float64,
            )
        )
        raw_increments = np.diff(values[origin - raw_lags : origin + 1])
        raw_inputs.append(
            np.concatenate((raw_increments, np.asarray([values[origin]], dtype=np.float64)))
        )
        keep_rows.append(row_index)

    selected = np.asarray(keep_rows, dtype=np.int64)
    geometry = bundle.representation("revisable_absolute").inputs[selected]
    path = np.vstack(descriptors)
    latest = np.vstack(latest_updates)
    candidate = np.concatenate((geometry, path), axis=1)
    inputs = {
        "geometry": geometry,
        "geometry_last_update": np.concatenate((geometry, latest), axis=1),
        "geometry_revision_path": candidate,
        "raw_matched": np.vstack(raw_inputs),
    }
    expected_widths = {
        "geometry": GEOMETRY_WIDTH,
        "geometry_last_update": LAST_UPDATE_WIDTH,
        "geometry_revision_path": PATH_INPUT_WIDTH,
        "raw_matched": PATH_INPUT_WIDTH,
    }
    for name, matrix in inputs.items():
        if matrix.shape != (selected.size, expected_widths[name]) or not np.all(
            np.isfinite(matrix)
        ):
            raise RuntimeError(f"invalid {name} design matrix")
        matrix.flags.writeable = False
    return RevisionPathDesign(
        mechanism=bundle.signal.mechanism,
        seed=bundle.signal.seed,
        origins=_readonly_int(bundle.origins[selected]),
        target_indices=_readonly_int(bundle.target_indices[selected]),
        targets=_readonly_float(bundle.targets[selected]),
        inputs=inputs,
    )


def _current_link_revision_increments(
    version: revisable_chain.WorkingVersion,
    histories: Mapping[int, list[tuple[int, float, float]]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    events: dict[int, NDArray[np.float64]] = {}
    counts = np.zeros(4, dtype=np.float64)
    for link_index, link in enumerate(version.links):
        history = histories.get(link.link_id, [])
        stop = bisect.bisect_right(history, (version.observed_at, float("inf"), float("inf")))
        counts[link_index] = stop
        for observed_at, update_theta, update_r in history[:stop]:
            vector = events.setdefault(observed_at, np.zeros(PATH_COORDINATES, dtype=np.float64))
            vector[2 * link_index] = update_theta
            vector[2 * link_index + 1] = update_r
    if not events:
        return np.empty((0, PATH_COORDINATES), dtype=np.float64), counts
    return np.vstack([events[key] for key in sorted(events)]), counts


def _revision_path_descriptor(
    increments: NDArray[np.float64], *, ages: NDArray[np.float64], counts: NDArray[np.float64]
) -> NDArray[np.float64]:
    signature = _path_signature_level_two(increments)
    if increments.shape[0] == 0:
        total_variation = np.zeros(PATH_COORDINATES, dtype=np.float64)
        energy = np.zeros(PATH_COORDINATES, dtype=np.float64)
        reversals = np.zeros(PATH_COORDINATES, dtype=np.float64)
    else:
        total_variation = np.sum(np.abs(increments), axis=0)
        energy = np.sum(increments**2, axis=0)
        reversals = np.asarray(
            [_sign_reversals(increments[:, column]) for column in range(PATH_COORDINATES)],
            dtype=np.float64,
        )
    descriptor = np.concatenate((signature, total_variation, energy, reversals, ages, counts))
    if descriptor.shape != (PATH_DESCRIPTOR_WIDTH,) or not np.all(np.isfinite(descriptor)):
        raise RuntimeError("invalid revision-path descriptor")
    return descriptor


def _path_signature_level_two(increments: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the first two signature levels of a piecewise-linear update path."""

    values = np.asarray(increments, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != PATH_COORDINATES:
        raise ValueError(f"increments must have shape (n, {PATH_COORDINATES})")
    first = np.zeros(PATH_COORDINATES, dtype=np.float64)
    second = np.zeros((PATH_COORDINATES, PATH_COORDINATES), dtype=np.float64)
    for increment in values:
        second += np.outer(first, increment) + 0.5 * np.outer(increment, increment)
        first += increment
    return np.concatenate((first, second.ravel()))


def _sign_reversals(values: NDArray[np.float64]) -> int:
    nonzero = np.asarray(values, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    if nonzero.size < 2:
        return 0
    return int(np.count_nonzero(nonzero[1:] * nonzero[:-1] < 0.0))


def sham_path_inputs(candidate: NDArray[np.float64], *, seed: int) -> NDArray[np.float64]:
    """Break path/target alignment while preserving geometry, width and marginals."""

    values = np.asarray(candidate, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != PATH_INPUT_WIDTH:
        raise ValueError("candidate has the wrong shape for the sham-path control")
    result = values.copy()
    permutation = np.random.default_rng(seed).permutation(values.shape[0])
    result[:, GEOMETRY_WIDTH:] = values[permutation, GEOMETRY_WIDTH:]
    return result


def run_kill_test(
    config_path: Path,
    *,
    output_root: Path | None = None,
    command_args: Sequence[str] = (),
) -> KillTestSummary:
    """Run alpha selection, untouched-suffix evaluation and the stopping gate."""

    config = load_config(config_path)
    root = Path(config.output_root) if output_root is None else output_root
    config_bytes = config_path.resolve().read_bytes()
    digest = hashlib.sha256(config_bytes).hexdigest()
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:10]}"
    run_dir = root / config.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    environment = _environment(config_path, command_args, digest)
    _write_json(run_dir / "config.json", config.raw)
    _write_json(run_dir / "environment.json", environment)

    designs: dict[tuple[str, int], RevisionPathDesign] = {}
    for mechanism in config.mechanisms:
        for seed in config.seeds:
            print(f"build {mechanism} seed={seed}", flush=True)
            signal = revisable_chain.generate_k7_signal(
                mechanism,
                seed=seed,
                n_points=config.n_points,
                noise_std=config.noise_std,
            )
            bundle = revisable_chain.build_k7_designs(
                signal,
                lambda_revision=config.lambda_revision,
                lambda_bend=config.lambda_bend,
            )
            designs[(mechanism, seed)] = build_revision_path_design(
                bundle, raw_lags=config.raw_lags
            )

    selected_alphas, selection_rows = _select_alphas(config, designs)
    metric_rows = _evaluate_test(config, designs, selected_alphas)
    ratios, gate = _evaluate_gate(config, metric_rows)
    _write_csv(run_dir / "selection.csv", selection_rows)
    _write_json(run_dir / "selection.json", selected_alphas)
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "ratios.csv", ratios)
    _write_json(run_dir / "gate.json", gate)
    (run_dir / "README.md").write_text(
        _result_markdown(config, selected_alphas, gate), encoding="utf-8", newline="\n"
    )
    environment["status"] = "complete"
    environment["finished_utc"] = datetime.now(UTC).isoformat()
    environment["elapsed_s"] = (time.perf_counter_ns() - started) / 1e9
    environment["gate_passed"] = bool(gate["passed"])
    _write_json(run_dir / "environment.json", environment)
    _write_manifest(run_dir)
    return KillTestSummary(run_id, run_dir, run_dir / "gate.json", bool(gate["passed"]))


def _select_alphas(
    config: KillTestConfig,
    designs: Mapping[tuple[str, int], RevisionPathDesign],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    means: dict[tuple[str, float], float] = {}
    for representation in REPRESENTATIONS:
        for alpha in config.alphas:
            scores: list[float] = []
            for key, design in designs.items():
                train = _split_mask(design, 0, config.train_end)
                validation_mask = _split_mask(design, config.train_end, config.validation_end)
                train_inputs = _inputs_for_split(design, representation, train, "train")
                validation_inputs = _inputs_for_split(
                    design, representation, validation_mask, "validation"
                )
                model = validation.fit_multi_ridge(train_inputs, design.targets[train], alpha=alpha)
                predictions = model.predict(validation_inputs)
                for column, horizon in enumerate(revisable_chain.HORIZONS):
                    rmse = _rmse(predictions[:, column], design.targets[validation_mask, column])
                    scale = max(float(np.std(design.targets[train, column])), 1e-12)
                    nrmse = rmse / scale
                    scores.append(nrmse)
                    rows.append(
                        {
                            "representation": representation,
                            "alpha": alpha,
                            "mechanism": key[0],
                            "seed": key[1],
                            "horizon": horizon,
                            "nrmse": nrmse,
                        }
                    )
            means[(representation, alpha)] = float(np.mean(scores))
    selected: dict[str, float] = {}
    for representation in REPRESENTATIONS:
        selected[representation] = min(
            config.alphas,
            key=lambda alpha: (means[(representation, alpha)], -alpha),
        )
    return selected, rows


def _evaluate_test(
    config: KillTestConfig,
    designs: Mapping[tuple[str, int], RevisionPathDesign],
    selected_alphas: Mapping[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (mechanism, seed), design in designs.items():
        fit = _split_mask(design, 0, config.validation_end)
        test = _split_mask(design, config.validation_end, config.test_end)
        for representation in REPRESENTATIONS:
            fit_inputs = _inputs_for_split(design, representation, fit, "fit")
            test_inputs = _inputs_for_split(design, representation, test, "test")
            model = validation.fit_multi_ridge(
                fit_inputs,
                design.targets[fit],
                alpha=selected_alphas[representation],
            )
            predictions = model.predict(test_inputs)
            for column, horizon in enumerate(revisable_chain.HORIZONS):
                rows.append(
                    {
                        "mechanism": mechanism,
                        "seed": seed,
                        "representation": representation,
                        "horizon": horizon,
                        "n_fit": int(np.count_nonzero(fit)),
                        "n_test": int(np.count_nonzero(test)),
                        "input_width": fit_inputs.shape[1],
                        "predictive_parameters": model.n_predictive_parameters,
                        "alpha": selected_alphas[representation],
                        "rmse": _rmse(predictions[:, column], design.targets[test, column]),
                    }
                )
    return rows


def _inputs_for_split(
    design: RevisionPathDesign,
    representation: str,
    mask: NDArray[np.bool_],
    split: str,
) -> NDArray[np.float64]:
    if representation != "geometry_sham_path":
        return np.asarray(design.inputs[representation][mask], dtype=np.float64)
    seed = _derived_seed(design.mechanism, design.seed, split)
    return sham_path_inputs(design.inputs["geometry_revision_path"][mask], seed=seed)


def _evaluate_gate(
    config: KillTestConfig, metrics: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    lookup = {
        (
            str(row["mechanism"]),
            int(row["seed"]),
            int(row["horizon"]),
            str(row["representation"]),
        ): float(row["rmse"])
        for row in metrics
    }
    ratio_rows: list[dict[str, object]] = []
    comparisons: dict[str, dict[str, object]] = {}
    for label, comparator in COMPARATORS.items():
        threshold = config.raw_ratio_max if label == "raw_matched" else config.path_ratio_max
        current: list[dict[str, object]] = []
        for mechanism in config.mechanisms:
            for seed in config.seeds:
                for horizon in revisable_chain.HORIZONS:
                    ratio = (
                        lookup[(mechanism, seed, horizon, "geometry_revision_path")]
                        / lookup[(mechanism, seed, horizon, comparator)]
                    )
                    row = {
                        "comparison": label,
                        "comparator": comparator,
                        "mechanism": mechanism,
                        "seed": seed,
                        "horizon": horizon,
                        "ratio": ratio,
                    }
                    current.append(row)
                    ratio_rows.append(row)
        seed_passes = _group_passes(current, "seed", threshold)
        mechanism_passes = _group_passes(current, "mechanism", threshold)
        horizon_passes = _group_passes(current, "horizon", threshold)
        global_ratio = _geometric_mean([float(row["ratio"]) for row in current])
        passed = (
            global_ratio <= threshold
            and seed_passes >= config.required_seed_passes
            and mechanism_passes >= config.required_mechanism_passes
            and horizon_passes >= config.required_horizon_passes
        )
        comparisons[label] = {
            "comparator": comparator,
            "threshold": threshold,
            "global_ratio": global_ratio,
            "seed_passes": seed_passes,
            "mechanism_passes": mechanism_passes,
            "horizon_passes": horizon_passes,
            "passed": passed,
        }
    return ratio_rows, {
        "candidate": "geometry_revision_path",
        "requirements": {
            "required_seed_passes": config.required_seed_passes,
            "required_mechanism_passes": config.required_mechanism_passes,
            "required_horizon_passes": config.required_horizon_passes,
        },
        "comparisons": comparisons,
        "passed": all(bool(result["passed"]) for result in comparisons.values()),
    }


def _group_passes(rows: Sequence[Mapping[str, object]], field: str, threshold: float) -> int:
    grouped: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(float(row["ratio"]))
    return sum(_geometric_mean(values) <= threshold for values in grouped.values())


def _split_mask(
    design: RevisionPathDesign, start_inclusive: int, end_exclusive: int
) -> NDArray[np.bool_]:
    mask = validation.endpoint_mask(
        design.target_indices,
        start_inclusive=start_inclusive,
        end_exclusive=end_exclusive,
    )
    if not np.any(mask):
        raise RuntimeError("a configured split contains no eligible origins")
    return mask


def _derived_seed(mechanism: str, seed: int, split: str) -> int:
    payload = f"revision-path-sham|{mechanism}|{seed}|{split}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _rmse(predicted: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(expected)) ** 2)))


def _geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise RuntimeError("ratios must be finite and positive")
    return float(np.exp(np.mean(np.log(array))))


def _result_markdown(
    config: KillTestConfig,
    selected_alphas: Mapping[str, float],
    gate: Mapping[str, object],
) -> str:
    comparisons = gate["comparisons"]
    if not isinstance(comparisons, Mapping):
        raise RuntimeError("gate comparisons are malformed")
    lines = [
        "# Revision-path kill test",
        "",
        f"Status: **{'passou' if gate['passed'] else 'não passou'}**.",
        "",
        "A candidata preserva a geometria revisada e acrescenta a trajetória completa de correções",
        "dos quatro elos atuais. A trajetória é resumida por assinatura de ordem 2, variação total,",
        "energia, reversões, idade e contagem de revisões.",
        "",
        "| Comparação | Razão global | Limiar | Seeds | Mecanismos | Horizontes | Passou |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label in COMPARATORS:
        result = comparisons[label]
        if not isinstance(result, Mapping):
            raise RuntimeError("gate comparison is malformed")
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{float(result['global_ratio']):.6f}",
                    f"{float(result['threshold']):.3f}",
                    str(result["seed_passes"]),
                    str(result["mechanism_passes"]),
                    str(result["horizon_passes"]),
                    "sim" if result["passed"] else "não",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Alphas selecionados exclusivamente na validação:",
            "",
            "```json",
            json.dumps(selected_alphas, indent=2, sort_keys=True),
            "```",
            "",
            f"Seeds novas: `{list(config.seeds)}`. Teste: `[{config.validation_end}, {config.test_end})`.",
            "",
            "Decisão: o projeto só continua por esta hipótese se todas as quatro comparações passarem.",
            "",
        )
    )
    return "\n".join(lines)


def _environment(config_path: Path, args: Sequence[str], digest: str) -> dict[str, object]:
    return {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *args],
        "config_path": str(config_path.resolve()),
        "config_sha256": digest,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args), check=False, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _write_manifest(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
        if path.name == "manifest.json" or not path.is_file():
            continue
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    _write_json(run_dir / "manifest.json", {"files": rows})


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_integer(item, name, minimum=0) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be real")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _real_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_real(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _readonly_float(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    result.flags.writeable = False
    return result


def _readonly_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64)
    result.flags.writeable = False
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured experiment and print its durable decision path."""

    args = _parse_args(argv)
    summary = run_kill_test(
        args.config,
        output_root=args.output_root,
        command_args=("experiments/11_revision_path_kill_test.py", "--config", str(args.config)),
    )
    print(f"run_id={summary.run_id}")
    print(f"run_dir={summary.run_dir}")
    print(f"gate={summary.gate_path}")
    print(f"passed={str(summary.passed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
