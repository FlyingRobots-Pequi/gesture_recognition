# Base leve e estável
FROM python:3.10-slim

# Evita prompts do apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependências de sistema para OpenCV/MediaPipe/ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copia apenas requirements primeiro (melhor aproveitamento de cache)
COPY requirements.txt .

# Instala as libs Python
# Torch CPU: se quiser forçar wheel CPU oficial, comente a linha de baixo e descomente as duas seguintes
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Porta de vídeo do Tello (11111) e controle (8889 UDP) – úteis se não usar --network=host
EXPOSE 11111/udp
EXPOSE 8889/udp

CMD ["python", "tello.py"]
