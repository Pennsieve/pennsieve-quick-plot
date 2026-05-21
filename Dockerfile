FROM python:3.12-slim

WORKDIR /app

# Install runtime deps (slim — heavy scientific deps live on EFS layer `quick-plot-stack`)
COPY processor/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY processor/ /app/processor/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH="/app"

# Dual-mode entrypoint: detects Lambda vs ECS at runtime
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command for ECS / local mode
CMD ["python", "-m", "processor.main"]
