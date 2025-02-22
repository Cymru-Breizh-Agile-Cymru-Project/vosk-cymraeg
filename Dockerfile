FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="jtrmal@apptek.com"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        make \
        automake \
        bzip2 \
        unzip \
        wget \
        libtool \
        git \
        python3 \
        zlib1g-dev \
        ca-certificates \
        gfortran \
        patch \
        sox \
        software-properties-common && \
        apt-add-repository multiverse && \
        apt-get update && \
        yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq --no-install-recommends\
            intel-mkl && \
    rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/kaldi-asr/kaldi.git /opt/kaldi
WORKDIR /opt/kaldi/tools
RUN git reset --hard 6f6139300b448e9ffc5abb04f2028928404be55c
RUN make -j $(nproc)

WORKDIR /opt/kaldi/src
RUN ./configure --shared --use-cuda=yes
RUN make depend -j $(nproc)
RUN make -j $(nproc)
RUN find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \;
RUN rm -rf /opt/kaldi/.git

WORKDIR /opt/kaldi/

# Install apt packages
# (only curl is required for UV, the rest are for the developer)
RUN apt update
RUN apt upgrade -y
RUN apt install -y \
    curl \
    fish \
    gawk \
    gcc \ 
    git \
    locales \
    make \
    neovim \
 	python2.7 \
    tar \
    wget

# Install en_US.UTF-8 (see issue #17)
RUN locale-gen en_US.UTF-8

# Force python to refer to python2
RUN ln -s python2 /usr/bin/python

ARG NAME="Preben Vangberg"
ARG AFFIL="Bangor University"
ARG EMAIL="prv21fgt@bangor.ac.uk"
ARG ADDRESS="Bangor, UK"

WORKDIR /opt/kaldi/tools
RUN bash install_srilm.sh "$NAME" "$AFFIL" "$EMAIL" "$ADDRESS"

# Setup the user profile (these can be modified as required)
ARG USERNAME=prv21fgt
ARG USER_UID=1001
ARG USER_GID=1001

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Change the default shell
RUN chsh -s /usr/bin/fish $USERNAME

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Download the config files
RUN git clone https://gitlab.com/prvInSpace/uniplexed-commafiles ~/.config

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Fix to get rid of Git warning
RUN git config --global --add safe.directory /opt/kaldi

# Set the workdir to the mounted volume
WORKDIR /workspaces/vosk

ENTRYPOINT /usr/bin/fish

