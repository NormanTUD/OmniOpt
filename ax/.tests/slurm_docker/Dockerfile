FROM debian:latest

# Umgebungsvariablen für nicht-interaktive Installation
ENV DEBIAN_FRONTEND=noninteractive

# System-Updates und benötigte Pakete installieren
RUN apt-get update
RUN apt-get install -y slurm-wlm vim munge mariadb-server
RUN rm -rf /var/lib/apt/lists/*

# Munge einrichten
#RUN mkdir -p /etc/munge /var/lib/munge /var/log/munge && \
#    chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge && \
#    chmod 0700 /etc/munge /var/lib/munge /var/log/munge && \
#    dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key && \
#    chown munge:munge /etc/munge/munge.key && \
#    chmod 0400 /etc/munge/munge.key

RUN rm /etc/munge/munge.key
RUN /usr/sbin/mungekey

# Slurm Konfiguration vorbereiten
COPY slurm.conf /etc/slurm/slurm.conf
RUN chown slurm:slurm /etc/slurm/slurm.conf

# MySQL initialisieren
#RUN mkdir -p /var/run/mysqld
#RUN chown -R mysql:mysql /var/run/mysqld
#RUN chmod -R 777 /var/run/mysqld

# Start-Skript für den Container
COPY internal_start.sh /internal_start.sh
RUN chown -R mysql:mysql /var/lib/mysql
RUN chmod +x /internal_start.sh

# Standardbefehl
CMD ["/internal_start.sh"]
