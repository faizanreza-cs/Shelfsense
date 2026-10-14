FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
COPY pyproject.toml README.md LICENSE ./
COPY shelfsense ./shelfsense
RUN uv pip install --system --no-cache ".[llm]"
# Generate the synthetic store and run every model at build time
RUN shelfsense build
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "shelfsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
