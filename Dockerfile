# The advisor blotter, one world per visitor, for Cloud Run or anywhere
# that runs a container. No database and no build step: the engine is
# pure Python and the UI is one HTML file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MERIDIAN_HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8080
CMD ["python", "-m", "meridian.api"]
