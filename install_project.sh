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
    cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    make install -j4
    popd

    pushd $HOME
    git clone git@github.com:acados/tera_renderer.git
    cd tera_renderer && cargo build --verbose --release
    cp $HOME/tera_renderer/target/release/t_renderer $HOME/acados/bin/t_renderer
    chmod +x $HOME/acados/bin/t_renderer
    popd
    
    # Fix macOS rpath issue: add rpath to acados libraries so @rpath resolves correctly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_name_tool -add_rpath "$HOME/acados/lib" "$HOME/acados/lib/libacados.dylib" 2>/dev/null || true
    fi
fi

uv pip install $HOME/acados/interfaces/acados_template