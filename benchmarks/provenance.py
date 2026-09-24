# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
"""Execution provenance that a third party needs to reproduce a run.

The fields recorded here are the ones that a run cannot reconstruct after the
fact: toolchain and driver versions, the identity of the model checkpoints that
were actually loaded, and the hash of the input media. Publication runs refuse
to start when a field that their selected suites depend on cannot be recorded,
so a run never silently produces evidence whose origin is unknown.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import warnings
from threading import Lock
from typing import Any

__all__ = [
    "UNKNOWN",
    "collect_provenance",
    "missing_required_provenance",
    "install_native_warning_recorder",
    "native_extension_runtime_state",
]

#: Marker for a field that could not be determined. It is never replaced by a
#: plausible default: unknown provenance must stay visible as unknown.
UNKNOWN = "unknown"
_warning_lock = Lock()
_observed_native_warnings: list[str] = []
_original_showwarning: Any = None


def install_native_warning_recorder() -> None:
    """Observe import-time native-extension warnings without fabricating text."""
    global _original_showwarning
    if _original_showwarning is not None:
        return
    _original_showwarning = warnings.showwarning

    def record_warning(
        message: Warning, category: type[Warning], filename: str,
        lineno: int, file: Any = None, line: str | None = None,
    ) -> None:
        text = str(message)
        if "tokenizers" in text and "GIL" in text:
            with _warning_lock:
                _observed_native_warnings.append(text)
        _original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = record_warning


def _run_version_command(argv: tuple[str, ...]) -> str:
    executable = shutil.which(argv[0])
    if executable is None:
        return UNKNOWN
    try:
        completed = subprocess.run(
            (executable,) + argv[1:],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    output = completed.stdout.strip() or completed.stderr.strip()
    if not output:
        return UNKNOWN
    return output.splitlines()[0].strip()


def _file_sha256(path: str) -> str:
    if not path or not os.path.isfile(path):
        return UNKNOWN
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(65536):
                digest.update(chunk)
    except OSError:
        return UNKNOWN
    return digest.hexdigest()


def _torch_provenance(device: str) -> dict[str, Any]:
    """Describe the Torch/CUDA stack as the process actually loaded it."""
    try:
        import torch
    except Exception:
        return {
            "torch_version": UNKNOWN,
            "cuda_runtime_version": UNKNOWN,
            "cudnn_version": UNKNOWN,
            "nvidia_driver_version": UNKNOWN,
            "gpu_name": UNKNOWN,
            "gpu_total_memory_bytes": UNKNOWN,
        }

    details: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda or UNKNOWN),
        "cudnn_version": UNKNOWN,
        "nvidia_driver_version": UNKNOWN,
        "gpu_name": UNKNOWN,
        "gpu_total_memory_bytes": UNKNOWN,
    }
    try:
        if torch.backends.cudnn.is_available():
            details["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception:
        pass

    if not device.startswith("cuda"):
        return details
    try:
        if not torch.cuda.is_available():
            return details
        index = 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        properties = torch.cuda.get_device_properties(index)
        details["gpu_name"] = str(properties.name)
        details["gpu_total_memory_bytes"] = int(properties.total_memory)
    except Exception:
        return details

    details["nvidia_driver_version"] = _run_version_command(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    )
    return details


#: Native extensions audited for free-threading support. Whether each declares
#: it is measured by ``_probe_free_threading_declaration``. Importing an
#: undeclared one asks CPython to re-enable the GIL; ``-Xgil=0`` overrides that,
#: and the run's recorded GIL checks prove which way it went.
_UNDECLARED_FREE_THREADING_MODULES: tuple[str, ...] = ("tokenizers",)


_FREE_THREADING_PROBE = """
import json, sys, warnings
warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import importlib
    importlib.import_module(sys.argv[1])
probe = getattr(sys, "_is_gil_enabled", None)
print(json.dumps({
    "gil_enabled_after_import": probe() if callable(probe) else None,
    "warnings": [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)],
}))
"""


def _probe_free_threading_declaration(module_name: str) -> dict[str, Any]:
    """Import a native module in a child interpreter without a GIL override.

    CPython re-enables the GIL, and warns, only for an extension that does not
    declare free-threading support. The child's observed GIL state therefore
    measures the declaration instead of asserting it.
    """
    environment = {key: value for key, value in os.environ.items() if key != "PYTHON_GIL"}
    try:
        completed = subprocess.run(
            (sys.executable, "-c", _FREE_THREADING_PROBE, module_name),
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
            check=False,
        )
    except Exception as error:
        return {"probe_error": repr(error)}
    if completed.returncode != 0:
        return {"probe_error": completed.stderr.strip()[-2000:]}
    try:
        import json

        result: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as error:
        return {"probe_error": f"unparseable probe output: {error!r}"}
    gil_after_import = result.get("gil_enabled_after_import")
    return {
        "probe_gil_enabled_after_import": gil_after_import,
        "probe_runtime_warnings": result.get("warnings", []),
        "declares_free_threading_support": (
            None if gil_after_import is None else not bool(gil_after_import)
        ),
    }


def native_extension_runtime_state() -> dict[str, Any]:
    """Report, for this process, which audited extensions were imported."""
    return {
        name: {
            "imported_in_benchmark_process": any(
                module == name or module.startswith(f"{name}.") for module in tuple(sys.modules)
            ),
            "observed_import_warnings": [
                text for text in tuple(_observed_native_warnings) if f"'{name}." in text or f"'{name}'" in text
            ],
        }
        for name in _UNDECLARED_FREE_THREADING_MODULES
    }


def _native_extension_provenance() -> dict[str, Any]:
    """Disclose audited extensions using measured, not asserted, evidence."""
    from importlib import metadata

    runtime_state = native_extension_runtime_state()
    disclosures: dict[str, Any] = {}
    for name in _UNDECLARED_FREE_THREADING_MODULES:
        try:
            version = metadata.version(name)
        except Exception:
            version = UNKNOWN
        entry: dict[str, Any] = {"version": version}
        if version != UNKNOWN:
            entry.update(_probe_free_threading_declaration(name))
        entry.update(runtime_state[name])
        disclosures[name] = entry
    return disclosures


def _model_provenance(model_ids: tuple[str, ...]) -> dict[str, Any]:
    """Resolve each model id to the exact revision the cache holds."""
    resolved: dict[str, Any] = {}
    for model_id in model_ids:
        entry: dict[str, Any] = {"revision": UNKNOWN, "config_sha256": UNKNOWN}
        try:
            from huggingface_hub import try_to_load_from_cache

            config_path = try_to_load_from_cache(model_id, "config.json")
            if isinstance(config_path, str):
                entry["config_sha256"] = _file_sha256(config_path)
                # A cached file lives under <repo>/snapshots/<revision>/<name>.
                parts = os.path.normpath(config_path).split(os.sep)
                if "snapshots" in parts:
                    revision_index = parts.index("snapshots") + 1
                    if revision_index < len(parts):
                        entry["revision"] = parts[revision_index]
        except Exception:
            pass
        resolved[model_id] = entry
    return resolved


def collect_provenance(
    selected_suites: tuple[str, ...],
    device: str,
    source_video: str,
    rtsp_base_url: str,
    model_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Collect every provenance field the selected suites can depend on."""
    provenance: dict[str, Any] = {
        "ffmpeg_version": _run_version_command(("ffmpeg", "-version")),
        "ffprobe_version": _run_version_command(("ffprobe", "-version")),
        "rtsp_base_url": rtsp_base_url,
    }
    provenance.update(_torch_provenance(device))
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    provenance["gil_enabled_after_imports"] = gil_probe() if callable(gil_probe) else UNKNOWN
    provenance["gil_xoption"] = getattr(sys, "_xoptions", {}).get("gil", UNKNOWN)
    provenance["python_gil_environment"] = os.environ.get("PYTHON_GIL", UNKNOWN)
    provenance["undeclared_free_threading_extensions"] = _native_extension_provenance()

    if "realworld-video" in selected_suites:
        provenance["source_video_path"] = source_video
        provenance["source_video_sha256"] = _file_sha256(source_video)
        provenance["models"] = _model_provenance(model_ids)

    return provenance


def missing_required_provenance(
    provenance: dict[str, Any],
    selected_suites: tuple[str, ...],
    device: str,
) -> tuple[str, ...]:
    """Name the provenance fields a publication run may not proceed without."""
    missing: list[str] = []

    def require(field: str) -> None:
        if provenance.get(field, UNKNOWN) == UNKNOWN:
            missing.append(field)

    if "realworld-video" not in selected_suites:
        return ()

    require("ffmpeg_version")
    require("ffprobe_version")
    require("source_video_sha256")
    require("torch_version")

    if device.startswith("cuda"):
        require("cuda_runtime_version")
        require("nvidia_driver_version")
        require("gpu_name")

    models: Any = provenance.get("models", {})
    if isinstance(models, dict):
        for model_id, entry in models.items():
            if not isinstance(entry, dict):
                missing.append(f"models.{model_id}")
                continue
            if entry.get("revision", UNKNOWN) == UNKNOWN:
                missing.append(f"models.{model_id}.revision")
            if entry.get("config_sha256", UNKNOWN) == UNKNOWN:
                missing.append(f"models.{model_id}.config_sha256")
    else:
        missing.append("models")

    return tuple(missing)
