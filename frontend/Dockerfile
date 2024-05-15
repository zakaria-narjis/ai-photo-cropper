# frontend/Dockerfile

FROM python:3.11.9-slim

RUN mkdir -p /app/frontend

COPY frontend/requirements.txt /app/frontend/requirements.txt
COPY frontend/pages/ /app/frontend/pages/

COPY frontend/Home.py /app/frontend
COPY frontend/user_session.py /app/frontend
COPY frontend/utils.py /app/frontend
COPY frontend/api_hundler.py /app/frontend

WORKDIR /app/frontend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]