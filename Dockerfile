FROM kaldiasr/kaldi

# Install apt packages
# (only curl is required for UV, the rest are for the developer)
RUN apt update
RUN apt upgrade -y
RUN apt install -y git wget make tar neovim fish gcc curl gawk

WORKDIR /opt/kaldi/tools
RUN bash install_srilm.sh

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

