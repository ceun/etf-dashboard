FROM python:3.11-slim

WORKDIR /app

COPY vue_app/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY etf_app.py /app/etf_app.py
COPY vue_app/backend /app/vue_app/backend

EXPOSE 8000
CMD ["uvicorn", "vue_app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
