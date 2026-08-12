.PHONY: help run build clean schemas

SERVICE_NAME ?= "pennsieve-quick-plot"
# Where the generated template/tool catalogs are vendored (see
# pennsieve-mcp/internal/tools/schemas/README.md).
MCP_SCHEMAS_DIR ?= ../pennsieve-mcp/internal/tools/schemas

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make build   - build the Docker image"
	@echo "make run     - run the processor locally via docker-compose"
	@echo "make clean   - remove output files"
	@echo "make schemas - regenerate the template/tool catalogs pennsieve-mcp embeds"

schemas:
	python3 -m processor.templates.generate_template_schema --out $(MCP_SCHEMAS_DIR)
	python3 -m processor.tools.generate_tools_schema --out $(MCP_SCHEMAS_DIR)

build:
	docker build -t $(SERVICE_NAME) .

run:
	docker-compose -f docker-compose.yml down --remove-orphans
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up --exit-code-from processor

clean:
	rm -f data/output/*
