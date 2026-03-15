# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging as py_logging
import os
import shutil
from pathlib import Path

import pytest
from transformers import logging as hf_logging
from collections import defaultdict

from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.logging_utils import logger

# Reduce noisy PyTorch C++ warning logs (e.g., torchvision op registration warnings)
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def qeff_models_clean_up(qeff_dir=QEFF_HOME):
    """
    Clean up QEFF models and cache.

    Args:
        qeff_dir: Can be a string (file/dir path), PosixPath, or list of strings/PosixPath objects
                 If a file path is provided, its parent directory will be deleted
    """
    if isinstance(qeff_dir, (str, Path)):
        paths = [qeff_dir]
    else:
        paths = qeff_dir

    for path in paths:
        try:
            path_str = str(path)
            if os.path.isfile(path_str):
                dir_to_delete = os.path.dirname(path_str)
                if os.path.exists(dir_to_delete):
                    shutil.rmtree(dir_to_delete)
                    print(f"\n.............Cleaned up {dir_to_delete}")
            elif os.path.isdir(path_str):
                if os.path.exists(path_str):
                    shutil.rmtree(path_str)
                    print(f"\n.............Cleaned up {path_str}")
        except Exception as e:
            print(f"\n.............Error cleaning up {path}: {e}")


@pytest.fixture
def manual_cleanup():
    """Fixture to manually trigger cleanup"""
    return qeff_models_clean_up


def pytest_sessionstart(session):
    logger.info("PYTEST Session Starting ...")

    # Suppress transformers warnings about unused weights when loading models with fewer layers
    hf_logging.set_verbosity_error()

    # Suppress noisy ONNX torchvision-missing warnings from torch exporter internals.
    py_logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(py_logging.ERROR)
    py_logging.getLogger("torch.onnx").setLevel(py_logging.ERROR)

    qeff_models_clean_up()


def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "llm_model: mark test as a pure LLM model inference test")
    config.addinivalue_line(
        "markers", "feature: mark test as a feature-specific test (SPD, sampler, prefix caching, LoRA, etc.)"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    group_stats = defaultdict(lambda: defaultdict(int))
    seen_status = set()
    seen_total = set()

    def _group_from_report(report):
        keywords = getattr(report, "keywords", {}) or {}
        if "llm_model" in keywords:
            return "llm_model"
        if "feature" in keywords:
            return "feature"
        return "unmarked"

    for status in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
        for report in terminalreporter.stats.get(status, []):
            nodeid = getattr(report, "nodeid", None)
            when = getattr(report, "when", "call")
            if not nodeid or when != "call":
                continue
            group = _group_from_report(report)

            status_key = (group, nodeid, status)
            if status_key in seen_status:
                continue
            seen_status.add(status_key)
            group_stats[group][status] += 1

            total_key = (group, nodeid)
            if total_key not in seen_total:
                seen_total.add(total_key)
                group_stats[group]["total"] += 1

    headers = ["group", "total", "passed", "failed", "skipped", "xfailed", "xpassed", "error"]
    rows = []
    order = ["llm_model", "feature", "unmarked"]
    for group in order:
        if group not in group_stats:
            continue
        rows.append([group] + [str(group_stats[group][name]) for name in headers[1:]])

    if not rows:
        return

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt(row):
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    terminalreporter.write_sep("-", "QEff Test Summary")
    terminalreporter.write_line(fmt(headers))
    terminalreporter.write_line("-+-".join("-" * w for w in widths))
    for row in rows:
        terminalreporter.write_line(fmt(row))

    xfailed_reports = [r for r in terminalreporter.stats.get("xfailed", []) if getattr(r, "when", "call") == "call"]
    failed_reports = [r for r in terminalreporter.stats.get("failed", []) if getattr(r, "when", "call") == "call"]

    if xfailed_reports:
        terminalreporter.write_sep("-", "Known Limitations (xfailed)")
        for report in xfailed_reports:
            reason = getattr(getattr(report, "longrepr", None), "reprcrash", None)
            reason_text = reason.message if reason and hasattr(reason, "message") else "expected failure"
            terminalreporter.write_line(f"- {report.nodeid}: {reason_text}")

    if failed_reports:
        terminalreporter.write_sep("-", "Failures")
        for report in failed_reports:
            terminalreporter.write_line(f"- {report.nodeid}")


def pytest_sessionfinish(session, exitstatus):
    inside_worker = getattr(session.config, "workerinput", None)
    if inside_worker is None:
        qeff_models_clean_up()
        logger.info("...PYTEST Session Ended.")
