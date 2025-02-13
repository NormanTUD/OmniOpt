FROM debian:latest

# Verhindere interaktive Abfragen bei apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update und Installation aller benötigten Pakete:
# - mariadb-server: für MariaDB
# - vim: Editor
# - supervisor: um beide Daemons im Container zu starten
RUN apt-get update && \
    apt-get install -y \
      mariadb-server \
      vim \
      supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Kopiere die slurm.conf aus dem Build-Kontext in den Container
# Achte darauf, dass die Datei im gleichen Ordner liegt wie dieses Dockerfile
COPY slurm.conf /etc/slurm-llnl/slurm.conf

# Kopiere die Supervisor-Konfiguration in den Container
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Exponiere den MariaDB-Port
EXPOSE 3306

# Starte Supervisor im Vordergrund
CMD ["/usr/bin/supervisord", "-n"]
