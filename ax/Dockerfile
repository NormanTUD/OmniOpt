FROM debian:bookworm

# Copy the PHP files to the container
COPY . /var/opt/omniopt/

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git

RUN bash /var/opt/omniopt/omniopt

#CMD ["apache2-foreground"]
