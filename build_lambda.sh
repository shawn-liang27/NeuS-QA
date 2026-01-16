#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Comprehensive Builder for Lambda Server
# Builds: CMake -> m4 -> Autoconf -> Automake -> Libtool -> GMP -> Carl -> Storm
# ==============================================================================

REPO_DIR="$(pwd)"
VENDORS_DIR="$REPO_DIR/vendors"
INSTALL_PREFIX="$VENDORS_DIR/install"
BUILD_THREADS="$(nproc)"

# --- Environment Setup ---
# Critical: Add install prefix to paths so we use our local tools immediately
export PATH="$INSTALL_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:${CMAKE_PREFIX_PATH:-}"
export CPATH="$INSTALL_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$VENDORS_DIR"
mkdir -p "$INSTALL_PREFIX"
cd "$VENDORS_DIR"

echo ">>> Installing dependencies to: $INSTALL_PREFIX"

# ------------------------------------------------------------------------------
# 0. Install CMake (System version is too old)
# ------------------------------------------------------------------------------
echo ">>> Installing CMake 3.29.3..."
if [ ! -d "cmake-3.29.3-linux-x86_64" ]; then
    wget -q https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.tar.gz
    tar -xf cmake-3.29.3-linux-x86_64.tar.gz
fi
cp -r cmake-3.29.3-linux-x86_64/* "$INSTALL_PREFIX/"
echo ">>> CMake installed."

# ------------------------------------------------------------------------------
# 1. Build m4 (Required by Autoconf & GMP)
# ------------------------------------------------------------------------------
echo ">>> Downloading and Building m4..."
M4_VER="1.4.19"
if [ ! -d "m4-$M4_VER" ]; then
    wget -q https://ftp.gnu.org/gnu/m4/m4-$M4_VER.tar.gz
    tar -xf m4-$M4_VER.tar.gz
fi
cd "m4-$M4_VER"
if [ ! -f "Makefile" ]; then
    ./configure --prefix="$INSTALL_PREFIX"
fi
make -j"$BUILD_THREADS"
make install
cd "$VENDORS_DIR"

# ------------------------------------------------------------------------------
# 1b. Build Autoconf (Required for CUDD)
# ------------------------------------------------------------------------------
echo ">>> Downloading and Building Autoconf..."
AUTOCONF_VER="2.71"
if [ ! -d "autoconf-$AUTOCONF_VER" ]; then
    wget -q https://ftp.gnu.org/gnu/autoconf/autoconf-$AUTOCONF_VER.tar.gz
    tar -xf autoconf-$AUTOCONF_VER.tar.gz
fi
cd "autoconf-$AUTOCONF_VER"
if [ ! -f "Makefile" ]; then
    ./configure --prefix="$INSTALL_PREFIX" M4="$INSTALL_PREFIX/bin/m4"
fi
make -j"$BUILD_THREADS"
make install
cd "$VENDORS_DIR"

# ------------------------------------------------------------------------------
# 1c. Build Automake (Required for CUDD)
# ------------------------------------------------------------------------------
echo ">>> Downloading and Building Automake..."
AUTOMAKE_VER="1.16.5"
if [ ! -d "automake-$AUTOMAKE_VER" ]; then
    wget -q https://ftp.gnu.org/gnu/automake/automake-$AUTOMAKE_VER.tar.gz
    tar -xf automake-$AUTOMAKE_VER.tar.gz
fi
cd "automake-$AUTOMAKE_VER"
if [ ! -f "Makefile" ]; then
    ./configure --prefix="$INSTALL_PREFIX"
fi
make -j"$BUILD_THREADS"
make install
cd "$VENDORS_DIR"

# ------------------------------------------------------------------------------
# 1d. Build Libtool (Required for CUDD)
# ------------------------------------------------------------------------------
echo ">>> Downloading and Building Libtool..."
LIBTOOL_VER="2.4.7"
if [ ! -d "libtool-$LIBTOOL_VER" ]; then
    wget -q https://ftp.gnu.org/gnu/libtool/libtool-$LIBTOOL_VER.tar.gz
    tar -xf libtool-$LIBTOOL_VER.tar.gz
fi
cd "libtool-$LIBTOOL_VER"
if [ ! -f "Makefile" ]; then
    ./configure --prefix="$INSTALL_PREFIX"
fi
make -j"$BUILD_THREADS"
make install
cd "$VENDORS_DIR"

echo ">>> Autotools stack installed successfully."

# ------------------------------------------------------------------------------
# 2. Build GMP
# ------------------------------------------------------------------------------
echo ">>> Downloading and Building GMP..."
GMP_VER="6.3.0"
if [ ! -d "gmp-$GMP_VER" ]; then
    wget -q https://gmplib.org/download/gmp/gmp-$GMP_VER.tar.xz
    tar -xf gmp-$GMP_VER.tar.xz
fi
cd "gmp-$GMP_VER"
if [ ! -f "Makefile" ]; then
    ./configure --prefix="$INSTALL_PREFIX" --enable-cxx --disable-static M4="$INSTALL_PREFIX/bin/m4"
fi
make -j"$BUILD_THREADS"
make install
cd "$VENDORS_DIR"

# ------------------------------------------------------------------------------
# 3. Build carl-storm
# ------------------------------------------------------------------------------
echo ">>> Building carl-storm..."
if [ ! -d "carl-storm" ]; then
  git clone https://github.com/moves-rwth/carl-storm
fi
cmake -S carl-storm -B carl-storm/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DGMP_DIR="$INSTALL_PREFIX" \
  -DUSE_CLN=OFF \
  -DUSE_GINAC=OFF
cmake --build carl-storm/build -j"$BUILD_THREADS" --target lib_carl
cmake --build carl-storm/build --target install

# ------------------------------------------------------------------------------
# 4. Build storm-stable
# ------------------------------------------------------------------------------
echo ">>> Building storm-stable..."
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
cmake --build storm-stable/build -j"$BUILD_THREADS"
cmake --build storm-stable/build --target install

# ------------------------------------------------------------------------------
# 5. Finalize
# ------------------------------------------------------------------------------
ENV_FILE="$REPO_DIR/activate_storm.sh"
cat <<EOF > "$ENV_FILE"
export PATH="$INSTALL_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:\${LD_LIBRARY_PATH:-}"
export CPATH="$INSTALL_PREFIX/include:\${CPATH:-}"
export CMAKE_PREFIX_PATH="$INSTALL_PREFIX:\${CMAKE_PREFIX_PATH:-}"
export STORM_DIR_HINT="$INSTALL_PREFIX"
export CARL_DIR_HINT="$INSTALL_PREFIX"
EOF

echo "========================================================="
echo "Build Complete!"
echo "1. Run: source activate_storm.sh"
echo "2. Run: uv sync"
echo "========================================================="