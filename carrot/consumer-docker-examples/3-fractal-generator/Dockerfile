FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir numpy pillow

# Create output directory
RUN mkdir -p /output

# Copy fractal generator script
COPY fractal.py /app/fractal.py
RUN chmod +x /app/fractal.py

WORKDIR /app

# Run the fractal generator
CMD ["python", "fractal.py"]
