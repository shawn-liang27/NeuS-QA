#!/bin/bash
# Run "source activate_storm_env.sh" to activate.

module load Boost/1.82.0-GCC-12.3.0
module load GMP/6.3.0-GCCcore-13.3.0

export INSTALL_PREFIX=$(pwd)

export PATH="$INSTALL_PREFIX/bin:$PATH"

export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:$CMAKE_PREFIX_PATH"
export CPATH="$INSTALL_PREFIX/include:$CPATH"
export LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:$INSTALL_PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH"

export storm_DIR="$INSTALL_PREFIX/lib/cmake/storm"

export SKBUILD_CMAKE_ARGS="-DSTORM_USE_CLN_EA=OFF -DSTORM_USE_CLN_RF=OFF -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX"

# 5. Now run the sync

echo ">>> Storm/Carl environment activated."
echo ">>> Binaries are in: $(pwd)/NeuS-QA/vendors/install/bin"
