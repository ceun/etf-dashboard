from typing import Literal

from pydantic import BaseModel, Field


Currency = Literal["CNY", "USD", "HKD", "JPY", "EUR"]
DataSource = Literal["SZ", "ZZ", "YH", "YHE"]


class TargetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    index_code: str = Field(min_length=1, max_length=40)
    etf_code: str | None = Field(default=None, max_length=40)
    data_source: DataSource
    asset_currency: Currency = "CNY"
    report_currency: Currency = "CNY"


class TargetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    etf_code: str | None = Field(default=None, max_length=40)
    data_source: DataSource | None = None
    asset_currency: Currency | None = None
    report_currency: Currency | None = None


class SyncResult(BaseModel):
    index_code: str
    written_rows: int
    status: Literal["success", "failure"]
    message: str = ""
