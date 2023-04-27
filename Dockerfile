FROM python:3.8.5

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8501

COPY requirements.txt .
COPY image_processing ./image_processing
COPY interpretability_utils ./interpretability_utils
COPY model_utils ./model_utils
COPY *.py .
COPY pages ./pages

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["streamlit", "run", "Main.py", "--server.port=8501", "--server.address=0.0.0.0"]