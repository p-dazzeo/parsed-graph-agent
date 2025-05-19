# Agent Guidelines for adk-agent-doc-test

## Commands
- Run the application: `python main.py`
- Run tests: `pytest`
- Type checking: `mypy .`
- Linting: `ruff check .`
- Code formatting: `ruff format .`

## Code Style Guidelines
- **Imports**: Standard lib first, third-party next, local modules last
- **Type Hints**: Use full typing annotations, use `|` for Union types
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Docstrings for all classes and functions
- **Error Handling**: Use try/except with specific exception types and logging
- **Logging**: Use the `logging` module with appropriate log levels
- **Environment**: Use dotenv for loading environment variables

## Project Structure
- Uses LangGraph for workflow orchestration
- Uses LangChain and OpenAI for LLM interactions