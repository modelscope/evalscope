FROM python:3.10-slim


# Create workspace directory and set ownership
WORKDIR /app


# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    jupyter \
    jupyter_client


CMD ["tail", "-f", "/dev/null"]
