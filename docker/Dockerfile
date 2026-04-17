FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="ASCLEPIUS Contributors"
LABEL description="ASCLEPIUS Biomedical Signal AI Framework"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN pip install -e . --no-deps

# Expose Streamlit port
EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "dashboard/app.py"]
