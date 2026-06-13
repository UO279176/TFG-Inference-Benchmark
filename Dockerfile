# Leer el .env para determinar qué CUDA usar
ARG CUDA_TARGET=new

# Esta imagen es específica para la Jetson Nano, que tiene CUDA 10.2 y solo puede utilizar hasta el Python 3.6
FROM ubuntu:18.04 AS base-old
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3-pip \
    python3-numpy \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==1.10.0 -f https://nvidia.box.com

WORKDIR /app
COPY requirements-old.txt ./requirements.txt

# Esta imagen es para sistemas con CUDA 12.6 y Python 3.10, concretamente para la Jetson Orin Nano y la Jetson Orin AGX
FROM ubuntu:22.04 AS base-new
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-numpy \
    sox \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY requirements-new.txt ./requirements.txt

# Usamos la imagen base correspondiente según el CUDA_TARGET
FROM base-${CUDA_TARGET} AS final

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/main.py ./src/
COPY src/data.py ./src/
COPY src/inference/ ./src/inference/
ENTRYPOINT ["python3", "src/main.py"]