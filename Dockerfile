# Author: Julia McCall
# Disclaimer: LLM was used in the initial setup of this file which was then adapted to our project.

FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work directory
WORKDIR /app

# Dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn psycopg2-binary

# Copy the project
COPY . /app/

# Entrypoint to run migrations and start server
COPY ./docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]