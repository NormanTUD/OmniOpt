FROM debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail zip python3-tk curl wget

ARG GetMyUsername
RUN adduser --disabled-password --gecos '' ${GetMyUsername}

WORKDIR /var/opt/omniopt

COPY ./omniopt ./omniopt
COPY ./.helpers.py ./.helpers.py
COPY ./.pareto.py ./.pareto.py
COPY ./.general.py ./.general.py
COPY ./.shellscript_functions.py ./.shellscript_functions.py
COPY ./.colorfunctions.py ./.colorfunctions.py
COPY ./.tpe.py ./.tpe.py
COPY ./requirements.txt ./requirements.txt
COPY ./test_requirements.txt ./test_requirements.txt

RUN ./omniopt --install

COPY ./.tests ./.tests
COPY ./.tools ./.tools
COPY ./.gui ./.gui
COPY ./ ./
