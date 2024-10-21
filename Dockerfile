FROM python:3.10-slim
RUN pwd

ENV PYTHONPATH=/opt/scripts
RUN mkdir -p /opt/scripts

COPY . /opt/emu-mps

RUN pip install /opt/emu-mps/ci/emu_base
RUN pip install /opt/emu-mps/ci/emu_mps
