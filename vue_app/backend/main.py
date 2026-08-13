import math
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .core_loader import UploadBuffer, core
from .schemas import SyncResult, TargetCreate, TargetUpdate


def require_admin(x_admin_token: str | None = Header(default=None)):
    expected = os.getenv("ADMIN_API_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_API_TOKEN is not configured")
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid administrator token")


def clean_number(value: Any):
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def frame_records(frame: pd.DataFrame):
    work = frame.copy()
    for column in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[column]):
            work[column] = work[column].dt.strftime("%Y-%m-%d")
    work = work.replace({np.nan: None, np.inf: None, -np.inf: None})
    return work.to_dict(orient="records")


def get_target_or_404(index_code: str):
    targets = core.load_targets_from_db()
    for target in targets.values():
        if target["index_code"] == core._normalize_index_code(index_code):
            return target
    raise HTTPException(status_code=404, detail="Target not found")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not core.DATABASE_URL:
        print("DATABASE_URL_POOLER or DATABASE_URL is not configured", flush=True)
    yield


app = FastAPI(
    title="ETF Long-term Valuation API",
    version="1.0.0",
    description="API for the Vue edition of the ETF dashboard.",
    lifespan=lifespan,
)

allowed_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "database_configured": bool(core.DATABASE_URL)}


@app.get("/api/targets")
def list_targets():
    return list(core.load_targets_from_db().values())


@app.get("/api/database/summary")
def database_summary():
    connection = core.get_db_connection()
    if not connection:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        frame = pd.read_sql(
            """
            SELECT p.index_code, t.name, t.etf_code, t.data_source,
                   t.asset_currency, COUNT(*) AS row_count,
                   MIN(p.date) AS first_date, MAX(p.date) AS latest_date
            FROM etf_prices p
            LEFT JOIN etf_targets t ON p.index_code = t.index_code
            GROUP BY p.index_code, t.name, t.etf_code, t.data_source, t.asset_currency
            ORDER BY t.name
            """,
            connection,
        )
        return frame_records(frame)
    finally:
        connection.close()


@app.get("/api/targets/{index_code}/analysis")
def target_analysis(
    index_code: str,
    tradition_start: date = Query(default=date(2008, 10, 31)),
    tradition_end: date = Query(default_factory=date.today),
    rolling_window: int = Query(default=1250, ge=20, le=5000),
    ma_window: int = Query(default=250, ge=2, le=2000),
    deviation_pct: float = Query(default=15, ge=1, le=100),
):
    target = get_target_or_404(index_code)
    frame, scaling_factor = core.load_from_db(target["index_code"])
    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail="No price data")
    if len(frame) < rolling_window + 10:
        raise HTTPException(status_code=422, detail=f"At least {rolling_window + 10} price rows are required")
    try:
        figure, result = core.compute_and_plot(
            frame,
            target["name"],
            deviation_pct,
            tradition_start,
            tradition_end,
            rolling_window,
            ma_window,
            scaling_factor,
        )
        plt.close(figure)
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    plot_frame = result.pop("plot_df")
    metrics = {key: clean_number(value) if not isinstance(value, str) else value for key, value in result.items()}
    return {"target": target, "metrics": metrics, "series": frame_records(plot_frame)}


@app.get("/api/comparison")
def comparison(
    tradition_start: date = Query(default=date(2008, 10, 31)),
    tradition_end: date = Query(default_factory=date.today),
    rolling_window: int = Query(default=1250, ge=20, le=5000),
    ma_window: int = Query(default=250, ge=2, le=2000),
    deviation_pct: float = Query(default=15, ge=1, le=100),
):
    targets = core.load_targets_from_db()
    frame = core.build_comparison(
        deviation_pct,
        targets,
        tradition_start,
        tradition_end,
        rolling_window,
        ma_window,
    )
    return frame_records(frame)


@app.post("/api/targets", dependencies=[Depends(require_admin)])
def create_target(payload: TargetCreate):
    normalized_index = core._normalize_index_code(payload.index_code)
    existing = core.load_targets_from_db()
    if any(item["index_code"] == normalized_index or item["name"] == payload.name.strip() for item in existing.values()):
        raise HTTPException(status_code=409, detail="Target name or index code already exists")
    if payload.data_source in {"ZZ", "YHE"}:
        raise HTTPException(status_code=422, detail="ZZ and YHE targets must be created through the import endpoint")
    saved = core.save_target_to_db(
        normalized_index,
        payload.name.strip(),
        etf_code=payload.etf_code,
        scaling_factor=1.0,
        data_source=payload.data_source,
        asset_currency=payload.asset_currency,
        report_currency=payload.report_currency,
    )
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save target")
    return get_target_or_404(normalized_index)


@app.patch("/api/targets/{index_code}", dependencies=[Depends(require_admin)])
def update_target(index_code: str, payload: TargetUpdate):
    current = get_target_or_404(index_code)
    saved = core.save_target_to_db(
        current["index_code"],
        payload.name or current["name"],
        etf_code=payload.etf_code if payload.etf_code is not None else current["etf_code"],
        scaling_factor=current["scaling_factor"],
        data_source=payload.data_source or current["data_source"],
        asset_currency=payload.asset_currency or current["asset_currency"],
        report_currency=payload.report_currency or current["report_currency"],
    )
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to update target")
    return get_target_or_404(current["index_code"])


@app.delete("/api/targets/{index_code}", dependencies=[Depends(require_admin)])
def delete_target(index_code: str):
    target = get_target_or_404(index_code)
    connection = core.get_db_connection()
    if not connection:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM etf_prices WHERE index_code=%s", (target["index_code"],))
        deleted_prices = cursor.rowcount
        cursor.execute("DELETE FROM etf_targets WHERE index_code=%s", (target["index_code"],))
        connection.commit()
        return {"index_code": target["index_code"], "deleted_prices": deleted_prices}
    except Exception as error:
        connection.rollback()
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        connection.close()


@app.post("/api/targets/import", dependencies=[Depends(require_admin)])
async def import_target(
    file: UploadFile = File(...),
    name: str = Form(...),
    index_code: str = Form(...),
    etf_code: str = Form(...),
    data_source: str = Form(...),
    asset_currency: str = Form("CNY"),
):
    source = core._normalize_data_source(data_source)
    if source not in {"ZZ", "YHE"}:
        raise HTTPException(status_code=422, detail="Import endpoint supports ZZ or YHE")
    content = await file.read()
    parsed, message = core.parse_upload_file(UploadBuffer(content, file.filename or "upload.csv"))
    if parsed is None or parsed.empty:
        raise HTTPException(status_code=422, detail=message)
    stitch = core.stitch_with_tickflow if source == "ZZ" else core.stitch_with_yahoo
    combined, scaling_factor, stitch_date, stitch_message = stitch(
        parsed,
        etf_code,
        asset_currency=asset_currency,
        report_currency="CNY",
    )
    if combined is None:
        raise HTTPException(status_code=422, detail=stitch_message)
    normalized_index = core._normalize_index_code(index_code)
    if not core.save_target_to_db(
        normalized_index,
        name.strip(),
        etf_code=etf_code,
        scaling_factor=scaling_factor,
        stitch_date=stitch_date,
        data_source=source,
        asset_currency=asset_currency,
        report_currency="CNY",
    ):
        raise HTTPException(status_code=500, detail="Failed to save target")
    written = core.save_prices_to_db(combined, normalized_index)
    return {
        "target": get_target_or_404(normalized_index),
        "parsed_rows": len(parsed),
        "written_rows": written,
        "parse_message": message,
        "stitch_message": stitch_message,
    }


@app.post("/api/targets/{index_code}/sync", response_model=SyncResult, dependencies=[Depends(require_admin)])
def sync_target(index_code: str):
    target = get_target_or_404(index_code)
    try:
        _, _, written = core.sync_target_data(target["index_code"])
        return SyncResult(index_code=target["index_code"], written_rows=int(written), status="success")
    except Exception as error:
        return SyncResult(index_code=target["index_code"], written_rows=0, status="failure", message=str(error))


@app.post("/api/sync-all", dependencies=[Depends(require_admin)])
def sync_all():
    results = []
    for target in core.load_targets_from_db().values():
        try:
            _, _, written = core.sync_target_data(target["index_code"])
            results.append(SyncResult(index_code=target["index_code"], written_rows=int(written), status="success"))
        except Exception as error:
            results.append(SyncResult(index_code=target["index_code"], written_rows=0, status="failure", message=str(error)))
    return {"results": results, "failures": sum(item.status == "failure" for item in results)}


@app.post("/api/macro/refresh", dependencies=[Depends(require_admin)])
def refresh_macro():
    pe_rows, pe_message = core.fetch_and_store_hs300_pe()
    bond_rows, bond_message = core.fetch_and_store_cn10y_yield()
    return {
        "pe": {"rows": pe_rows, "message": pe_message},
        "bond": {"rows": bond_rows, "message": bond_message},
    }


@app.get("/api/rotation")
def rotation(
    asset_a: str,
    asset_b: str,
    k_multiplier: float = Query(default=1.5, ge=0, le=10),
    n_slots: int = Query(default=10, ge=1, le=100),
    erp_step: float = Query(default=0.5, gt=0, le=10),
):
    pe = core.load_macro_from_db("hs300_pe")
    bond = core.load_macro_from_db("cn10y_yield")
    frame_a, _ = core.load_from_db(asset_a)
    frame_b, _ = core.load_from_db(asset_b)
    if pe.empty or bond.empty:
        raise HTTPException(status_code=422, detail="Macro data is missing; refresh macro data first")
    if frame_a is None or frame_b is None:
        raise HTTPException(status_code=422, detail="Selected asset data is missing")
    erp = core.compute_erp(pe, bond, k_multiplier=k_multiplier)
    result = core.backtest_erp_rotation(erp, frame_a, frame_b, n_slots=n_slots, erp_step=erp_step)
    if not result:
        raise HTTPException(status_code=422, detail="Not enough overlapping data")
    timeline = result.pop("t_df")
    return {"metrics": {key: clean_number(value) for key, value in result.items()}, "series": frame_records(timeline)}
