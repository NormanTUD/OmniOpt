FROM debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

WORKDIR /var/opt/omniopt

COPY ./omniopt ./omniopt
COPY ./requirements.txt ./requirements.txt
COPY ./test_requirements.txt ./test_requirements.txt

RUN ./omniopt --install

COPY ./.tests ./.tests
COPY ./.tools ./.tools
COPY ./.gui ./.gui
COPY ./ ./
