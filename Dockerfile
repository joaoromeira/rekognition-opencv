# Pull base image
FROM jjanzic/docker-python3-opencv

# Set dir
WORKDIR /app

COPY . /app

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install --upgrade pip

# Install dependencies
RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["bash", "scripts/start-dev.sh"]