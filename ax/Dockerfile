FROM debian:bookworm

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail

COPY .shellscript_functions /.shellscript_functions
RUN bash /.shellscript_functions
RUN rm /.shellscript_functions

COPY . /var/opt/omniopt/
RUN bash .tests/example_network/install.sh

#CMD ["apache2-foreground"]
