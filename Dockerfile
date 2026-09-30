FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY kb kb
COPY static static

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# uvloop + httptools — самый быстрый цикл событий и HTTP-парсер для uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
