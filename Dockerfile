FROM python:3.12-slim

WORKDIR /app

# The processor's own runtime deps.
COPY processor/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The scientific stack the generated plotting scripts import, in a folder of its own
# that only their child process puts on PYTHONPATH (processor/executor.py), so the
# main process stays lean. Byte-compiled here: Lambda's filesystem is read-only.
COPY processor/stack-requirements.txt /app/stack-requirements.txt
RUN pip install --no-cache-dir --target /opt/plot-stack -r stack-requirements.txt \
    && python -m compileall -q /opt/plot-stack

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
