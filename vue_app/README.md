# ETF Dashboard Vue Edition

这是原 Streamlit 项目的 Vue 3 + FastAPI 版本。它复用现有 Python 数据逻辑，因此回归、汇率、ETF 拼接和每日同步口径与旧版一致。

## 功能

- 单标的价格、长期回归、滚动回归、MA 和 Z-Score 图表。
- 全市场偏离度、CAGR、回归范围和置信区间比较。
- `SZ`、`ZZ`、`YH`、`YHE` 四条数据链路。
- CSV/XLSX 历史文件导入与 ETF 后复权拼接。
- 单标的同步、全量同步和标的删除。
- 沪深 300 ERP 股债轮动回测。
- 桌面和移动端响应式界面。

## 本地开发

要求：Python 3.11、Node.js 22、PostgreSQL/Supabase 连接串。

### 后端

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r vue_app\backend\requirements.txt
$env:DATABASE_URL_POOLER="postgresql://..."
$env:ADMIN_API_TOKEN="your-token"
uvicorn vue_app.backend.main:app --reload --port 8000
```

接口文档：`http://localhost:8000/docs`。

### 前端

```powershell
cd vue_app\frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

打开 `http://localhost:5173`。

如果后端配置了 `ADMIN_API_TOKEN`，请在 `.env.local` 中设置相同的 `VITE_ADMIN_API_TOKEN`。生产环境不应把管理员令牌直接放入公开前端；应接入登录和服务端会话。当前令牌方式只适合个人项目或内网部署。

## Docker 启动

```powershell
cd vue_app
Copy-Item .env.example .env
docker compose up --build
```

打开 `http://localhost:8080`。

## API 结构

```text
GET    /api/health
GET    /api/targets
GET    /api/targets/{index_code}/analysis
GET    /api/comparison
GET    /api/database/summary
POST   /api/targets
POST   /api/targets/import
PATCH  /api/targets/{index_code}
DELETE /api/targets/{index_code}
POST   /api/targets/{index_code}/sync
POST   /api/sync-all
POST   /api/macro/refresh
GET    /api/rotation
```

## 每日同步

新项目可直接运行：

```powershell
python -m vue_app.backend.sync_all
```

确认 Vue 版稳定后，可将 GitHub Actions 中的 `python sync_all.py` 改为上述命令。

## 迁移原则

- Vue 不直接访问数据库，也不保存数据库密码。
- FastAPI 负责数据库、行情、上传和分析。
- `combined_close` 继续作为统一分析价格。
- 当前通过兼容加载器复用 `etf_app.py` 的核心部分。后续可逐步将核心函数拆到独立 `services` 模块，再让 Streamlit 和 FastAPI 同时调用。
