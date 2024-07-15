FROM debian:bookworm

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ENV root_venv_dir=/
COPY .shellscript_functions /.shellscript_functions
RUN bash /.shellscript_functions
RUN rm /.shellscript_functions

COPY .tests/example_network/install.sh /.test_install.sh
RUN bash /.test_install.sh
RUN rm /.test_install.sh

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

COPY ./ /var/opt/omniopt/
COPY ./.tests /var/opt/omniopt/.tests
COPY ./.tools /var/opt/omniopt/.tools
COPY ./.gui /var/opt/omniopt/.gui

WORKDIR /var/opt/omniopt
