FROM debian:bookworm

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv xterm git uuid-runtime

ENV VIRTUAL_ENV=
RUN python3 -m venv "~/.omniax_$(uname -m)_$(python3 --version | sed -e 's# #_#g')"
ENV PATH="~/.omniax_$(uname -m)_$(python3 --version | sed -e 's# #_#g')/bin:$PATH"
RUN pip3 install wheel
RUN pip3 install setuptools>=61
RUN pip3 install numpy<=1.26.4
RUN pip3 install graphql-core>=2.0
RUN pip3 install packaging<=20.1
RUN pip3 install python-dateutil<=2.8.1
RUN pip3 install Rich
RUN pip3 install scikit-learn
RUN pip3 install submitit
RUN pip3 install ax
RUN pip3 install ax-platform
RUN pip3 install matplotlib
RUN pip3 install Pyarrow
RUN pip3 install textual
RUN pip3 install tqdm
RUN pip3 install seaborn
RUN pip3 install art
RUN pip3 install pyvista
RUN pip3 install catsay
RUN pip3 install sixel
RUN pip3 install pytz
RUN pip3 install tzlocal
RUN pip3 install rich-argparse

COPY . /var/opt/omniopt/
RUN bash /var/opt/omniopt/.shellscript_functions
RUN bash .tests/example_network/install.sh

#CMD ["apache2-foreground"]
