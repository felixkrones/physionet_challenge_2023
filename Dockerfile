FROM python:3.10.1

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="Felix Krones felix.krones@oii.ox.ac.uk"
LABEL maintainer="Ben Walker ben.walker@ladeck.com"

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt update
RUN apt install -y libsndfile1-dev
RUN wget "https://download.pytorch.org/models/densenet121-a639ec97.pth"

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
