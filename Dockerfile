FROM debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ENV install_tests=1
ENV root_venv_dir=/
COPY .shellscript_functions /.shellscript_functions
COPY .colorfunctions.sh /.colorfunctions.sh
COPY requirements.txt /requirements.txt
COPY test_requirements.txt /test_requirements.txt
RUN bash /.shellscript_functions

COPY .tests/example_network/install.sh /.test_install.sh
RUN bash /.test_install.sh
RUN rm /.test_install.sh
RUN rm /.shellscript_functions

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

COPY ./ /var/opt/omniopt/
COPY ./.tests /var/opt/omniopt/.tests
COPY ./.tools /var/opt/omniopt/.tools
COPY ./.gui /var/opt/omniopt/.gui

WORKDIR /var/opt/omniopt
