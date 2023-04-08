FROM python:latest
# FROM ubuntu:latest

WORKDIR /app

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --upgrade pip

COPY ./requirements.txt /app
RUN pip install -v --no-cache-dir --upgrade -r requirements.txt
# RUN conda install --file requirements.txt

COPY . /app

CMD ["python", "model_api.py"]  