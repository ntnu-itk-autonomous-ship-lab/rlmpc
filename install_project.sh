#!/bin/bash

# Sync the main dependencies
uv sync

# Install acados
pushd $HOME
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

popd

uv pip install $HOME/acados/interfaces/acados_template