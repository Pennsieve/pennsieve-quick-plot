FROM python:3.12-slim

WORKDIR /app

# Install runtime deps (slim — heavy scientific deps live on EFS layer `quick-plot-stack`)
COPY processor/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY processor/ /app/processor/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH="/app"

# Point matplotlib's config + cache dir at /tmp/matplotlib so it doesn't try
# to write to ~/.config/matplotlib (read-only on Lambda; the failed mkdir +
# fallback adds ~1.5s + a misleading WARNING on every cold start). /tmp is
# writable on both Lambda and ECS, and the font cache persists for the
# warm container's lifetime — subsequent invocations skip the
# `generated new fontManager` step entirely.
ENV MPLCONFIGDIR="/tmp/matplotlib"

# Dual-mode entrypoint: detects Lambda vs ECS at runtime
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command for ECS / local mode
CMD ["python", "-m", "processor.main"]
