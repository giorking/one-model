FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ="Asia/Shanghai"
ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

RUN sed -i s/mirrors.163.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN apt-get update && apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libpostproc-dev libeigen3-dev libtbb-dev libgtk2.0-dev pkg-config liblapack-dev libopencv-dev -y
RUN apt-get install --no-install-recommends -y python3-dev libportaudio2 libasound-dev git python3 python3-pip make g++ ffmpeg && rm -rf /var/lib/apt/lists/*
RUN apt-get update&& apt-get install python-is-python3 -y

RUN pip config set global.index-url https://mirror.baidu.com/pypi/simple

# COPY requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /opt/product
ADD dist/segment-anything /opt/product/segment-anything
ADD dist/one_model /opt/product/one_model
RUN cd /opt/product/segment-anything && pip install -e .
RUN cd /opt/product/one_model && pip install -e .

WORKDIR /opt/product
