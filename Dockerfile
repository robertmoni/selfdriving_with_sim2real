FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04


RUN apt-get update -y && apt-get install -y software-properties-common && apt-get update -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN add-apt-repository ppa:deadsnakes/ppa


RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get install -y --no-install-recommends \
  bash \
  screen \
  git \
  make \
  python3.6 \
  python3-pip \
  python3-setuptools \
  python3-setuptools-git \
  python3.6-dev \
  xvfb \
  x11-utils\
  ffmpeg \
  wget \
  libgtk2.0-dev \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libfontconfig1 \
  libxrender1 \
  python3.6-tk 
RUN cd /usr/bin && rm python && ln -s python3.6 python

RUN python --version 

RUN apt-get install -y --no-install-recommends build-essential nvidia-cuda-toolkit

RUN python -m pip install --upgrade \
   pip \
   setuptools

RUN pip --version

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get install -y libxtst6 libxi6 libgtk2.0-0 libidn11 libglu1-mesa

COPY . /selfdriving_with_sim2real
# RUN python -c 'import torch; assert torch.cuda.is_available(), "cuda is not available." '

WORKDIR /selfdriving_with_sim2real
RUN rm -rf gym-duckietown/
RUN git clone --branch v6.1.25 --single-branch --depth 1 https://github.com/duckietown/gym-duckietown.git gym-duckietown/

WORKDIR /selfdriving_with_sim2real/gym-duckietown/
RUN pip install -e .

WORKDIR /selfdriving_with_sim2real
RUN cp -R maps/* gym-duckietown/src/gym_duckietown/maps
RUN cp -R maps/* 

RUN cp .screenrc /root
RUN apt-get install locales -y && locale-gen en_US.UTF-8

RUN chmod 777 -R /selfdriving_with_sim2real
WORKDIR /app

RUN chmod -R uo+rwx /app

WORKDIR /selfdriving_with_sim2real
CMD tail -f /dev/null
