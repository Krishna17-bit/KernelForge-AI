from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_dark_table(df: pd.DataFrame, height: int = 320) -> None:
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
    html = safe_df.to_html(index=False, escape=True)
    st.markdown(f"<div class='dark-table-scroll' style='max-height:{height}px;'>{html}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "") -> str:
    cls = "status-pill"
    if kind == "ok":
        cls += " pill-ok"
    elif kind == "warn":
        cls += " pill-warn"
    elif kind == "danger":
        cls += " pill-danger"
    return f"<span class='{cls}'>{text}</span>"


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        val = float(x)  # type: ignore[arg-type]
        if math.isfinite(val):
            return val
    except Exception:
        pass
    return default


def write_json(path: Path, payload: dict) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
