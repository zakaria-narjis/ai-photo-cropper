# backend/Dockerfile

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /app/backend

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
    
RUN apt-get install ffmpeg libsm6 libxext6  -y    
COPY backend/requirements.txt /app/backend/requirements.txt 
COPY backend/comp_cropping/ /app/backend/comp_cropping/
COPY backend/api.py /app/backend
COPY backend/clipcrop/ /app/backend/clipcrop/

WORKDIR /app/backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8080","--reload"]
