# Repository Instructions

## Project

## Technical Stack

- Python 3.12
- uv
- pytest
- ruff
- pyrefly

## Repository Structure

- `src/llm-faithfulness` - scripts for demonstrating llm faithfulness checks
- `tests` - unit and integration tests
- `openspec` - openspec artifacts
- `article` - article drafts

## Development Commands

- Install packages: `uv sync`
- Add a new package: `uv add`
- Run unit tests: `uv run pytest`
- Run typechecker: `uv run pyrefly`
- Run linter: `uvx --from ruff==0.16.10 ruff check .`

## Testing Requirements

- Create focused tests

## Security and Privacy

- Do not log access tokens or API keys
