# Dockerfile, Image, Container

# Dockerfile
# A blueprint for building Docker images

# Image
# A template for running containers

# Container
# The actual running process where we have a package project running

FROM python:3.11.4-bookworm

# Set the working directory
WORKDIR /app

ADD script.py .

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "script.py", "--server.port=8501", "--server.address=0.0.0.0"]