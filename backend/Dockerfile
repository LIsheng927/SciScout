FROM python:3.11-slim

WORKDIR /app

# 先只拷贝依赖清单再装依赖,这样代码变了不用重新下载依赖包(Docker层缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
