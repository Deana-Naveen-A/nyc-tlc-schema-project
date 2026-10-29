FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY contracts/ ./contracts/

RUN mkdir -p /app/results

CMD ["python", "src/contract_enforced_pipeline.py"]