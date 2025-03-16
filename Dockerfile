FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV USERNAME=mpi
ENV PASSWORD=mpi

RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev
