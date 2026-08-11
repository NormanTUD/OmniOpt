FROM debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ENV install_tests=1
ENV root_venv_dir=/
COPY .colorfunctions.sh /.colorfunctions.sh
COPY .shellscript_functions.py /.shellscript_functions.py
COPY requirements.txt /requirements.txt
COPY test_requirements.txt /test_requirements.txt
RUN python3 -c "import sys; sys.path.insert(0, '/'); from shellscript_functions import setup_environment; raise SystemExit(setup_environment())"

COPY .tests/example_network/install.py /.test_install.py
RUN python3 /.test_install.py
RUN rm /.test_install.py
RUN rm /.shellscript_functions.py

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

COPY ./ /var/opt/omniopt/
COPY ./.tests /var/opt/omniopt/.tests
COPY ./.tools /var/opt/omniopt/.tools
COPY ./.gui /var/opt/omniopt/.gui

WORKDIR /var/opt/omniopt
