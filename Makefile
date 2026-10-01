.PHONY: hello setup lint test

RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
MAGENTA=\033[0;35m
CYAN=\033[0;36m
RESET=\033[0m

hello:
	@echo "${MAGENTA}Hello, $$(whoami)!${RESET}"
	@echo "${GREEN}Current Time:${RESET}\t\t${YELLOW}$$(date)${RESET}"
	@echo "${GREEN}Working Directory:${RESET}\t${YELLOW}$$(pwd)${RESET}"
	@echo "${GREEN}Shell:${RESET}\t\t\t${YELLOW}$$(echo $$SHELL)${RESET}"
	@echo "${GREEN}Terminal:${RESET}\t\t${YELLOW}$$(echo $$TERM)${RESET}"


lint:
	@echo "Running linter..."
	@uv run pre-commit run --all-files
	@echo "Done."

setup:
	@echo "Installing Python dependencies..."
	@uv sync
	@uv run pre-commit install
	@echo "Downloading models..."
	@uv run python scripts/download_models.py

test:
	@uv run pytest -q
