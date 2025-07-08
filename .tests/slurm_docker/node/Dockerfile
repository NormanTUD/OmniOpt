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
    sudo vim bash git zip 2to3 curl xterm whiptail python3-dev php-cli unzip python3-lib2to3 python3-toolz php-mbstring bc uuid-runtime python3.10-venv zsh curl uuid-runtime xterm whiptail

RUN apt-get install -y adwaita-icon-theme bash-completion dh-exec fontconfig freeipmi-common gir1.2-atk-1.0 gir1.2-freedesktop gir1.2-gdkpixbuf-2.0 gir1.2-gtk-2.0 gir1.2-harfbuzz-0.0 gir1.2-pango-1.0 gtk-update-icon-cache hdf5-helpers hicolor-icon-theme humanity-icon-theme icu-devtools libaec-dev libaec0 libatk1.0-0 libatk1.0-data libatk1.0-dev libavahi-client3 libavahi-common-data libavahi-common3 libb64-0d libblkid-dev libbrotli-dev libcairo-gobject2 libcairo-script-interpreter2 libcairo2 libcairo2-dev libcups2 libcurl4-openssl-dev libdatrie-dev libdatrie1 libdbi1 libdeflate-dev libevent-2.1-7 libevent-core-2.1-7 libevent-dev libevent-extra-2.1-7 libevent-openssl-2.1-7 libevent-pthreads-2.1-7 libffi-dev libfontconfig-dev libfontconfig1-dev libfreeipmi-dev libfreeipmi17 libfreetype-dev libfreetype6-dev libfribidi-dev libgail-common libgail18 libgdk-pixbuf-2.0-0 libgdk-pixbuf-2.0-dev libgdk-pixbuf2.0-bin libgdk-pixbuf2.0-common libgfortran5 libglib2.0-bin libglib2.0-dev libglib2.0-dev-bin libgraphite2-3 libgraphite2-dev libgtk2.0-0 libgtk2.0-bin libgtk2.0-common libgtk2.0-dev libharfbuzz-dev libharfbuzz-gobject0 libharfbuzz-icu0 libharfbuzz0b libhdf5-103-1 libhdf5-cpp-103-1 libhdf5-dev libhdf5-fortran-102 libhdf5-hl-100 libhdf5-hl-cpp-100 libhdf5-hl-fortran-100 libhttp-parser-dev libhttp-parser2.9 libice-dev libicu-dev libipmimonitoring-dev libipmimonitoring6 libjansson4 libjbig-dev libjpeg-dev libjpeg-turbo8-dev libjpeg8-dev libjson-c-dev libjwt-dev libjwt0 liblua5.3-0 liblua5.3-dev liblz4-dev liblzma-dev libmount-dev libmunge-dev libncurses-dev libpam0g-dev libpango-1.0-0 libpango1.0-dev libpangocairo-1.0-0 libpangoft2-1.0-0 libpangoxft-1.0-0 libpcre16-3 libpcre2-16-0 libpcre2-32-0 libpcre2-dev libpcre2-posix3 libpcre3-dev libpcre32-3 libpcrecpp0v5 libperl-dev libpixman-1-0 libpixman-1-dev libpmix-dev libpmix2 libpng-dev libpng-tools libpthread-stubs0-dev librdkafka++1 librdkafka-dev librdkafka1 libreadline-dev librrd-dev librrd8 librsvg2-2 librsvg2-common libselinux1-dev libsepol-dev libsm-dev libssl-dev libsz2 libthai-data libthai-dev libthai0 libtiff-dev libtiffxx5 libtool-bin libx11-dev libxau-dev libxcb-render0 libxcb-render0-dev libxcb-shm0-dev libxcb1-dev libxcomposite-dev libxcursor-dev libxcursor1 libxdamage-dev libxdamage1 libxdmcp-dev libxext-dev libxfixes-dev libxft-dev libxi-dev libxinerama-dev libxml2-utils libxrandr-dev libxrender-dev libyaml-dev man2html-base pango1.0-tools ubuntu-mono uuid-dev x11proto-dev xorg-sgml-doctools xtrans-dev

RUN useradd -m admin -s /usr/bin/bash -d /home/admin \
 && echo "admin:admin" | chpasswd \
 && adduser admin sudo \
 && echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
RUN useradd -m slurm -s /usr/bin/bash -d /home/slurm \
 && echo "slurm:slurm" | chpasswd

RUN echo "#!/bin/bash" >> /bin/sq
RUN echo "squeue" >> /bin/sq
RUN chmod +x /bin/sq

RUN mkdir /var/run/sshd \
 && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
 && chmod -x /etc/update-motd.d/* \
 && rm -f /etc/legal

RUN cd /tmp \
 && wget https://download.schedmd.com/slurm/slurm-23.11.7.tar.bz2
RUN cd /tmp \
 && tar -xaf slurm-23.11.7.tar.bz2
RUN cd /tmp \
 && cd slurm-23.11.7 \
 && mk-build-deps -t "apt-get -y" -i debian/control \
 && debuild -b -uc -us
RUN cd /tmp \
 && ARCH=$(dpkg --print-architecture) \
 && dpkg --install slurm-smd_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-client_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-slurmd_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-dev_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-libpmi0_23.11.7-1_${ARCH}.deb \
 && dpkg --install slurm-smd-libpmi2-0_23.11.7-1_${ARCH}.deb \
 && rm -rf /tmp/slurm* \
 && rm -rf /var/lib/apt/lists/*

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

COPY munge.key /etc/munge/
COPY slurm.conf /etc/slurm/
COPY cgroup.conf /etc/slurm/
COPY docker-entrypoint.sh /etc/slurm/

RUN chown munge:munge etc/munge/munge.key \
 && mkdir /var/spool/slurmd \
 && chown slurm:slurm /var/spool/slurmd \
 && chmod 755 /var/spool/slurmd \
 && chown slurm:slurm /etc/slurm \
 && chown slurm:slurm /etc/slurm/*.conf \
 && chmod 600 /etc/munge/munge.key \
 && chmod 775 /etc/slurm \
 && chmod 775 /etc/slurm/*.conf

RUN systemctl enable slurmd

COPY ssh /home/admin/.ssh

RUN chown -R admin:admin /home/admin/.ssh \
 && chmod -R 700 /home/admin/.ssh

EXPOSE 22 6817 6818 6819

ENV PATH="/bin:${PATH}"

RUN echo '#!/bin/bash' > /bin/sacct && \
	echo '' >> /bin/sacct && \
	echo 'job_ids=()  # Job-IDs initialisieren' >> /bin/sacct && \
	echo '' >> /bin/sacct && \
	echo '# Argumente parsen' >> /bin/sacct && \
	echo 'while [[ $# -gt 0 ]]; do' >> /bin/sacct && \
	echo '    case "$1" in' >> /bin/sacct && \
	echo '        -j)  # Job-IDs sammeln' >> /bin/sacct && \
	echo '            shift' >> /bin/sacct && \
	echo '            job_ids+=("$1")' >> /bin/sacct && \
	echo '            ;;' >> /bin/sacct && \
	echo '        *)  # Alle anderen Optionen ignorieren' >> /bin/sacct && \
	echo '            ;;' >> /bin/sacct && \
	echo '    esac' >> /bin/sacct && \
	echo '    shift' >> /bin/sacct && \
	echo 'done' >> /bin/sacct && \
	echo '' >> /bin/sacct && \
	echo '# Header für die Ausgabe' >> /bin/sacct && \
	echo 'echo "JobID|State|NodeList"' >> /bin/sacct && \
	echo '' >> /bin/sacct && \
	echo '# Durch die Job-IDs iterieren' >> /bin/sacct && \
	echo 'for job_id in "${job_ids[@]}"; do' >> /bin/sacct && \
	echo '    # Überprüfen, ob der Job noch in squeue ist' >> /bin/sacct && \
	echo '    job_info=$(squeue -j "$job_id" --noheader -o "%i %T %N")' >> /bin/sacct && \
	echo '' >> /bin/sacct && \
	echo '    if [[ -n "$job_info" ]]; then' >> /bin/sacct && \
	echo '        # Job ist noch aktiv -> Status aus squeue nehmen' >> /bin/sacct && \
	echo "        job_state=\$(echo "\$job_info" | awk '{print \$2}')" >> /bin/sacct && \
	echo "        node_list=\$(echo "\$job_info" | awk '{print \$3}')" >> /bin/sacct && \
	echo '    else' >> /bin/sacct && \
	echo '        # Job nicht mehr in squeue -> als COMPLETED betrachten' >> /bin/sacct && \
	echo '        job_state="COMPLETED"' >> /bin/sacct && \
	echo '        node_list="UNKNOWN"' >> /bin/sacct && \
	echo '    fi' >> /bin/sacct && \
	echo '' >> /bin/sacct && \
	echo '    # Ausgabe im Parsable2-Format' >> /bin/sacct && \
	echo '    echo "$job_id|$job_state|$node_list"' >> /bin/sacct && \
	echo '    echo "$job_id.0|$job_state|$node_list"' >> /bin/sacct && \
	echo 'done' >> /bin/sacct && \
	chmod +x /bin/sacct

ENTRYPOINT ["/etc/slurm/docker-entrypoint.sh"]
