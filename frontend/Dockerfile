# frontend/Dockerfile

FROM python:3.11.9-slim

RUN mkdir -p /app/frontend

COPY requirements.txt /app/frontend/requirements.txt
COPY pages/ /app/frontend/pages/

COPY Home.py /app/frontend
COPY user_session.py /app/frontend
COPY utils.py /app/frontend
COPY api_hundler.py /app/frontend

WORKDIR /app/frontend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]