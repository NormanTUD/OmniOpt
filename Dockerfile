FROM debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

COPY ./ /var/opt/omniopt/
COPY ./.tests /var/opt/omniopt/.tests
COPY ./.tools /var/opt/omniopt/.tools
COPY ./.gui /var/opt/omniopt/.gui

WORKDIR /var/opt/omniopt

RUN ./omniopt --help
