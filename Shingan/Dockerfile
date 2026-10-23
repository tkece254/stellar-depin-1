FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY tests ./tests
RUN pip install --no-cache-dir '.[dev]'

ENTRYPOINT ["shingan"]
