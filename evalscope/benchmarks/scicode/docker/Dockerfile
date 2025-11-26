FROM python:3.11-slim

COPY docker_requirements.txt .
RUN pip install --no-cache-dir -r docker_requirements.txt

WORKDIR /workspace

COPY process_data.py test_util.py test_data.h5 /workspace/
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
