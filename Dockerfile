FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-plain-generic \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    duckdb==1.5.3 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    pyarrow==16.1.0

ENV PYTHONPATH=/app/src

CMD ["python", "src/01_build_dataset.py"]
