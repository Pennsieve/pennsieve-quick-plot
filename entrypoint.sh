#!/bin/sh
# Dual-mode entrypoint: detects Lambda vs ECS runtime and branches accordingly.
# AWS_LAMBDA_RUNTIME_API is set by the Lambda service, absent on ECS / local.

if [ -n "$AWS_LAMBDA_RUNTIME_API" ]; then
    # Lambda mode: start the Runtime Interface Client with our handler
    exec python -m awslambdaric processor.handler.handler
else
    # ECS / local mode: execute the default CMD
    exec "$@"
fi
