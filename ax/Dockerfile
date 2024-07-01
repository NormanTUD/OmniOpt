FROM debian:bookworm

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime

ENV VIRTUAL_ENV=
RUN python3 -m venv "~/.omniax_$(uname -m)_$(python3 --version | sed -e 's# #_#g')"
ENV PATH="~/.omniax_$(uname -m)_$(python3 --version | sed -e 's# #_#g')/bin:$PATH"
RUN pip3 --break-system-packages install wheel
RUN pip3 --break-system-packages install setuptools>=61
RUN pip3 --break-system-packages install numpy<=1.26.4
RUN pip3 --break-system-packages install graphql-core>=2.0
RUN pip3 --break-system-packages install packaging<=20.1
RUN pip3 --break-system-packages install python-dateutil<=2.8.1
RUN pip3 --break-system-packages install Rich
RUN pip3 --break-system-packages install scikit-learn
RUN pip3 --break-system-packages install submitit
RUN pip3 --break-system-packages install ax
RUN pip3 --break-system-packages install ax-platform
RUN pip3 --break-system-packages install matplotlib
RUN pip3 --break-system-packages install Pyarrow
RUN pip3 --break-system-packages install textual
RUN pip3 --break-system-packages install tqdm
RUN pip3 --break-system-packages install seaborn
RUN pip3 --break-system-packages install art
RUN pip3 --break-system-packages install pyvista
RUN pip3 --break-system-packages install catsay
RUN pip3 --break-system-packages install sixel
RUN pip3 --break-system-packages install pytz
RUN pip3 --break-system-packages install tzlocal
RUN pip3 --break-system-packages install rich-argparse

COPY . /var/opt/omniopt/
RUN bash /var/opt/omniopt/.shellscript_functions
RUN bash .tests/example_network/install.sh

#CMD ["apache2-foreground"]
