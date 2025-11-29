#!/bin/bash

# Sync the main dependencies
uv sync

# Ensure rlmpc is installed in editable mode (in case uv sync didn't do it)
uv pip install -e .

# Install acados if not already installed
if [ ! -d "$HOME/acados" ]; then
    pushd $HOME
    git clone https://github.com/acados/acados.git
    cd acados
    git submodule update --recursive --init
    mkdir -p build
    cd build
    cmake -DACADOS_WITH_QPOASES=ON ..
    make install -j4
    popd
fi

uv pip install $HOME/acados/interfaces/acados_template