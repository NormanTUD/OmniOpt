FROM debian:bookworm

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime whiptail

COPY .shellscript_functions /.shellscript_functions
RUN bash /.shellscript_functions
RUN rm /.shellscript_functions

COPY .tests/example_network/install.sh /.test_install.sh
RUN bash /.test_install.sh
RUN rm /.test_install.sh

COPY . /var/opt/omniopt

#CMD ["apache2-foreground"]
