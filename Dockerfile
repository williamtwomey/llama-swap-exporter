FROM python:3.12-slim

# The exporter itself only needs prometheus_client; all HTTP is stdlib urllib,
# deliberately, to keep this image small and its CVE surface near zero.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY llamaswap_exporter.py /app/llamaswap_exporter.py
WORKDIR /app

# Nothing here touches the host, GPUs or the filesystem — it is a JSON poller.
RUN useradd --system --uid 10001 --no-create-home exporter
USER 10001

ENV LLAMASWAP_URL=http://127.0.0.1:8080 \
    EXPORTER_PORT=9820 \
    POLL_INTERVAL=15 \
    PYTHONUNBUFFERED=1

EXPOSE 9820
ENTRYPOINT ["python", "/app/llamaswap_exporter.py"]
