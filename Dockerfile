FROM debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ENV install_tests=1
COPY requirements.txt /requirements.txt
COPY test_requirements.txt /test_requirements.txt
COPY .shellscript_functions.py /tmp/shellscript_functions.py
RUN cd /tmp && python3 shellscript_functions.py setup_environment && rm -f /tmp/shellscript_functions.py

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

COPY ./ /var/opt/omniopt/
COPY ./.tests /var/opt/omniopt/.tests
COPY ./.tools /var/opt/omniopt/.tools
COPY ./.gui /var/opt/omniopt/.gui

WORKDIR /var/opt/omniopt
