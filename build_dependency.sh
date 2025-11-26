#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(pwd)"
VENDORS_DIR="$REPO_DIR/vendors"
INSTALL_PREFIX="$VENDORS_DIR/install"

mkdir -p "$VENDORS_DIR"
cd "$VENDORS_DIR"

# carl-storm
cd "$VENDORS_DIR"
if [ ! -d "carl-storm" ]; then
  git clone https://github.com/moves-rwth/carl-storm
fi
cmake -S carl-storm -B carl-storm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DUSE_CLN=OFF \
  -DUSE_GINAC=OFF
cmake --build carl-storm/build -j"$(nproc)" --target lib_carl
cmake --build carl-storm/build --target install

# storm-stable
if [ ! -d "storm-stable" ]; then
  git clone --branch stable --depth 1 --recursive https://github.com/moves-rwth/storm.git storm-stable
fi
cmake -S storm-stable -B storm-stable/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
  -DSTORM_DEVELOPER=OFF \
  -DSTORM_LOG_DISABLE_DEBUG=ON \
  -DSTORM_PORTABLE=ON \
  -DSTORM_USE_SPOT_SHIPPED=ON \
  -DSTORM_USE_CLN_EA=OFF \
  -DSTORM_USE_CLN_RF=OFF
cmake --build storm-stable/build -j"$(nproc)"
cmake --build storm-stable/build --target install

export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export STORM_DIR_HINT="$INSTALL_PREFIX"
export CARL_DIR_HINT="$INSTALL_PREFIX"
unset CMAKE_ARGS || true


ENV_FILE="$REPO_DIR/activate_storm_env.sh"

cat <<EOF > "$ENV_FILE"
#!/bin/bash
# Auto-generated environment file for Storm/Carl
# Run "source $(basename "$ENV_FILE")" to activate.

export PATH="$INSTALL_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:\${LD_LIBRARY_PATH:-}"
export CPATH="$INSTALL_PREFIX/include:\${CPATH:-}"
export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:\${CMAKE_PREFIX_PATH:-}"

# Hints for other CMake projects finding these libs
export STORM_DIR_HINT="$INSTALL_PREFIX"
export CARL_DIR_HINT="$INSTALL_PREFIX"

echo ">>> Storm/Carl environment activated."
echo ">>> Binaries are in: $INSTALL_PREFIX/bin"
EOF

chmod +x "$ENV_FILE"

echo "=============================================================================="
echo "BUILD COMPLETE"
echo "To use the installed software, run the following command:"
echo "    source $ENV_FILE"
echo ""
echo "=============================================================================="