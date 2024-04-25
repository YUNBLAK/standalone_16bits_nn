FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "CNN_MAIN.py"]
