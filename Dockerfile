FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
#source: https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md

ARG DEBIAN_FRONTEND=noninteractive

#works 

RUN apt-get update -y && apt-get install -y software-properties-common && apt-get update -y
RUN apt-get install --reinstall ca-certificates
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    sudo \
    openssh-server \
    screen \
    cmake \
    vim \
    gcc \
    wget \
    htop \
    unzip \
    libc-dev\
    git \
    bzip2 \
    python3-pip \
    python3-setuptools \
    python3-setuptools-git \
    python3-dev \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
	
	
# Build and install nvtop
RUN apt-get update && apt-get install -y cmake libncurses5-dev git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /work/*

RUN apt-get update && apt-get install -y --no-install-recommends


# Expose ports for tensorboardand jupyter
EXPOSE 7022
EXPOSE 7023



# Install and configure ssh server
# https://docs.docker.com/engine/examples/running_ssh_service/
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server vim nano htop xauth
RUN echo 'PermitRootLogin yes\nSubsystem sftp internal-sftp\nX11Forwarding yes\nX11UseLocalhost no' > /etc/ssh/sshd_config
EXPOSE 22
RUN groupadd sshgroup
RUN useradd -ms /bin/bash -g sshgroup david

RUN echo 'root:self_driving42' | chpasswd

# add user permission on "/home/general/" path
WORKDIR /home/general
USER root


RUN echo "cd /home/general"


RUN mkdir /var/run/sshd
CMD ["/usr/sbin/sshd", "-D"]

#Setup ssh server
#RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 david

#RUN usermod -aG sudo david

#RUN service ssh start


#EXPOSE 22

#CMD ["/usr/sbin/sshd","-D"]
