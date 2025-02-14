FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y \
    build-essential \
    git \
    mariadb-server \
    munge \
    vim \
    wget \
    devscripts \
    equivs \
    fakeroot \
    libbpf-dev \
    libdbus-1-dev \
    libhwloc-dev \
    openssh-server \
    sudo \
 && cd /tmp \
 && wget https://download.schedmd.com/slurm/slurm-23.11.7.tar.bz2 \
 && tar -xaf slurm-23.11.7.tar.bz2 \
 && cd slurm-23.11.7 \
 && mk-build-deps -t "apt-get -y" -i debian/control \
 && debuild -b -uc -us \
 && cd .. \
 && ARCH=$(dpkg --print-architecture) \
 && dpkg --install slurm-smd_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-client_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-dev_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-libpmi0_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-libpmi2-0_23.11.7-1_${ARCH}.deb \
 && rm -rf /tmp/slurm* \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin \
 && echo "admin:admin" | chpasswd \
 && adduser admin sudo \
 && echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN useradd -m slurm -s /usr/bin/bash -d /home/slurm \
 && echo "slurm:slurm" | chpasswd

RUN mkdir /var/run/sshd \
 && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
 && chmod -x /etc/update-motd.d/* \
 && rm -f /etc/legal

RUN apt-get install -y vim bash git

RUN echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "# deb-src http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "# deb-src http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "# deb-src http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "# deb-src http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "" >> /etc/apt/sources.list && \
    echo "deb http://archive.canonical.com/ubuntu/ jammy partner" >> /etc/apt/sources.list && \
    echo "# deb-src http://archive.canonical.com/ubuntu/ jammy partner" >> /etc/apt/sources.list

RUN apt-get update -y

RUN apt-get install -y zip
RUN apt-get install -y 2to3
RUN apt-get install -y curl
RUN apt-get install -y xterm
RUN apt-get install -y whiptail
RUN apt-get install -y python3-dev
RUN apt-get install -y php-cli 
RUN apt-get install -y unzip
RUN apt-get install -y python3-lib2to3 
RUN apt-get install -y python3-toolz 
RUN apt-get install -y php-mbstring
RUN apt-get install -y bc
RUN apt-get install -y uuid-runtime
RUN apt-get install -y python3.10-venv
RUN apt-get install -y zsh

RUN echo "#!/bin/bash" >> /bin/sq
RUN echo "squeue" >> /bin/sq
RUN chmod +x /bin/sq

RUN echo '#!/bin/bash' > /bin/sacct
RUN echo '' >> /bin/sacct
RUN echo 'job_ids=()  # Job-IDs initialisieren' >> /bin/sacct
RUN echo '' >> /bin/sacct
RUN echo '# Argumente parsen' >> /bin/sacct
RUN echo 'while [[ $# -gt 0 ]]; do' >> /bin/sacct
RUN echo '    case "$1" in' >> /bin/sacct
RUN echo '        -j)  # Job-IDs sammeln' >> /bin/sacct
RUN echo '            shift' >> /bin/sacct
RUN echo '            job_ids+=("$1")' >> /bin/sacct
RUN echo '            ;;' >> /bin/sacct
RUN echo '        *)  # Alle anderen Optionen ignorieren' >> /bin/sacct
RUN echo '            ;;' >> /bin/sacct
RUN echo '    esac' >> /bin/sacct
RUN echo '    shift' >> /bin/sacct
RUN echo 'done' >> /bin/sacct
RUN echo '' >> /bin/sacct
RUN echo '# Header für die Ausgabe' >> /bin/sacct
RUN echo 'echo "JobID|State|NodeList"' >> /bin/sacct
RUN echo '' >> /bin/sacct
RUN echo '# Durch die Job-IDs iterieren' >> /bin/sacct
RUN echo 'for job_id in "${job_ids[@]}"; do' >> /bin/sacct
RUN echo '    # Überprüfen, ob der Job noch in squeue ist' >> /bin/sacct
RUN echo '    job_info=$(squeue -j "$job_id" --noheader -o "%i %T %N")' >> /bin/sacct
RUN echo '' >> /bin/sacct
RUN echo '    if [[ -n "$job_info" ]]; then' >> /bin/sacct
RUN echo '        # Job ist noch aktiv -> Status aus squeue nehmen' >> /bin/sacct
RUN echo "        job_state=\$(echo "\$job_info" | awk '{print \$2}')" >> /bin/sacct
RUN echo "        node_list=\$(echo "\$job_info" | awk '{print \$3}')" >> /bin/sacct
RUN echo '    else' >> /bin/sacct
RUN echo '        # Job nicht mehr in squeue -> als COMPLETED betrachten' >> /bin/sacct
RUN echo '        job_state="COMPLETED"' >> /bin/sacct
RUN echo '        node_list="UNKNOWN"' >> /bin/sacct
RUN echo '    fi' >> /bin/sacct
RUN echo '' >> /bin/sacct
RUN echo '    # Ausgabe im Parsable2-Format' >> /bin/sacct
RUN echo '    echo "$job_id|$job_state|$node_list"' >> /bin/sacct
RUN echo '    echo "$job_id.0|$job_state|$node_list"' >> /bin/sacct
RUN echo 'done' >> /bin/sacct
RUN chmod +x /bin/sacct

RUN echo 'for i in $(squeue --me | grep -v JOBID | sed -e "s#^\s*##" -e "s#\s.*##"); do scancel $i; done' >> /bin/scancel_all
RUN chmod +x /bin/scancel_all


COPY munge.key /etc/munge/
COPY slurm.conf /etc/slurm/
COPY cgroup.conf /etc/slurm/
COPY docker-entrypoint.sh /etc/slurm/

RUN chown munge:munge etc/munge/munge.key \
 && chown slurm:slurm /etc/slurm \
 && chown slurm:slurm /etc/slurm/*.conf \
 && chmod 600 /etc/munge/munge.key \
 && chmod 775 /etc/slurm \
 && chmod 775 /etc/slurm/*.conf

COPY ssh /home/admin/.ssh

RUN chown -R admin:admin /home/admin/.ssh \
 && chmod -R 700 /home/admin/.ssh

WORKDIR /home/admin

EXPOSE 22 8888

ENV USER admin
ENV SHELL bash

#RUN mkdir -p /oo/
#RUN git clone https://github.com/NormanTUD/OmniOpt.git /oo/
#RUN bash /oo/ax/.shellscript_functions

ENTRYPOINT ["/etc/slurm/docker-entrypoint.sh"]
