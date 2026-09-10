FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY apex ./apex
RUN pip install --no-cache-dir .
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["apex", "serve", "--host", "0.0.0.0", "--port", "8080"]
