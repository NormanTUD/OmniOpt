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

# Pre-create the test framework venv so `python3 .tests/main` inside
# the container doesn't have to `pip install` 70+ packages at runtime.
# This is the same venv that the autodeps framework creates under
# `~/.omniax_venvs/Python_${py_version}/${arch}/`, and the test runner
# keys off the same Python version the build image ships with.
RUN PY_VERSION=$(python3 --version | sed -e 's# #_#g') && \
    ARCH=$(uname -m) && \
    VENV_DIR="$HOME/.omniax_venvs/${PY_VERSION}/${ARCH}" && \
    mkdir -p "$(dirname "$VENV_DIR")" && \
    python3 -m venv "$VENV_DIR" && \
    "$VENV_DIR/bin/pip" --disable-pip-version-check install \
        -r requirements.txt -r test_requirements.txt && \
    # Cache the hashes so ensure_dependencies() skips re-install.
    md5sum requirements.txt | awk '{print $1}' > "$VENV_DIR/hash_main" && \
    md5sum test_requirements.txt | awk '{print $1}' > "$VENV_DIR/hash_test"


