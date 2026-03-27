# CPU-compatible task with LIVE PROGRESS LOGGING
# Shows real-time execution proof that Docker is running

FROM python:3.10-slim

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Copy Python task with verbose logging
COPY task.py /task.py

CMD ["python3", "-u", "/task.py"]
