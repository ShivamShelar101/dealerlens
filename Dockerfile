# DealerLens API + app. The LLM runs on the host (MLX on Apple Silicon) and is reached via host.docker.internal.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn python-multipart lightgbm numpy pandas requests \
    scikit-learn joblib streamlit tabulate
COPY . .
ENV DEALERLENS_LLM_URL=http://host.docker.internal:8081/v1
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
