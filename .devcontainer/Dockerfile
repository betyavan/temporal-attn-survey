# Fast layer on top of `make docker` (~1s). Fixes HOME=/workspaces conflict with devcontainer mounts.
FROM temporal-attn-survey:latest

ARG USERNAME=vscode
ENV USERNAME=${USERNAME}

RUN useradd -m -d /workspaces -s /bin/bash "$USERNAME" && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /root

USER ${USERNAME}