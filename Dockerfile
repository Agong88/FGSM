FROM tensorflow/tensorflow:2.14.0

RUN pip install --no-cache-dir \
    matplotlib \
    pillow \
    numpy \
    keras \
    h5py

RUN apt-get update && \
    apt-get install -y fonts-noto-cjk && \
    apt-get clean

WORKDIR /app

COPY main.py .

CMD ["python", "main.py"]
