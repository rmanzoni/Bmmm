#!/bin/bash
# Build Hammer v2 against CMSSW_16_0_8's own Python, so that the ntuplizer can
# `import hammer` inside the release without a conda environment.
#
#   cmsenv                       # CMSSW_16_0_8, el9_amd64_gcc13
#   bash install_hammer.sh       # ~15 min
#   source hammer_env.sh
#   python3 -c "import hammer; print(hammer.__file__)"
#
# Why from source and not conda-forge: the conda package is the 1.4.x series and
# ships its own Python; we need the v2 API (snake_case bindings, set_ff_eigenvectors)
# built against Python 3.9.14 + the release's numpy, or the bindings import but
# the array marshalling is wrong.
set -euo pipefail

PREFIX=${PREFIX:-/work/manzoni/hammer}
SRC=${SRC:-${PREFIX}/src}
BUILD=${BUILD:-${PREFIX}/build}
REPO=${REPO:-https://gitlab.com/mpapucci/Hammer.git}
TAG=${TAG:-master}          # pin a tag once a version is chosen; record it in the card
JOBS=${JOBS:-8}

if [ -z "${CMSSW_BASE:-}" ]; then
    echo "[FATAL] run cmsenv first: Hammer must be built against the release's Python." >&2
    exit 1
fi

echo "#### CMSSW_BASE $CMSSW_BASE"
echo "#### python     $(python3 --version 2>&1)   $(which python3)"
echo "#### gcc        $(gcc -dumpversion)"
echo "#### cmake      $(cmake --version | head -1)"
echo "#### cython     $(cython --version 2>&1)"
echo "#### prefix     $PREFIX"

# -- Boost ------------------------------------------------------------------
# CMake >= 3.30 no longer provides the FindBoost module (policy CMP0167), and
# Hammer's find_package(Boost 1.75.0) therefore lands in config mode: it wants
# an upstream BoostConfig.cmake, not a header directory. CMSSW ships boost as a
# scram external, so resolve it from there rather than from the system.
BOOST_BASE=${BOOST_BASE:-$(scram tool tag boost BOOST_BASE 2>/dev/null || true)}
if [ -z "$BOOST_BASE" ]; then
    echo "[FATAL] could not resolve boost from scram. Set BOOST_BASE by hand." >&2
    exit 1
fi
# Hammer asks for the `thread` component. In config mode BoostConfig.cmake
# resolves each component through its own boost_<lib>-config.cmake, so the
# top-level config file alone is not enough -- check for the component too and
# drop to the find-module route if it is missing.
BOOST_CMAKE_DIR=$(ls -d "$BOOST_BASE"/lib*/cmake/Boost-* 2>/dev/null | head -1 || true)
BOOST_THREAD_DIR=$(ls -d "$BOOST_BASE"/lib*/cmake/boost_thread-* 2>/dev/null | head -1 || true)
BOOST_FLAGS=()
if [ -n "$BOOST_CMAKE_DIR" ] && [ -n "$BOOST_THREAD_DIR" ]; then
    echo "#### boost      $BOOST_BASE"
    echo "####            config mode via $BOOST_CMAKE_DIR (+ thread component)"
    BOOST_FLAGS+=(-DBoost_DIR="$BOOST_CMAKE_DIR")
else
    # No BoostConfig.cmake in the external: fall back to CMake's own FindBoost
    # module by putting CMP0167 back to OLD. This only works if Hammer's
    # find_package(Boost ...) is a plain call -- if it says CONFIG or NO_MODULE
    # explicitly, the module is never consulted and the config file is mandatory.
    echo "#### boost      $BOOST_BASE"
    if [ -z "$BOOST_CMAKE_DIR" ]; then
        echo "####            no BoostConfig.cmake -> FindBoost module (CMP0167=OLD)"
    else
        echo "####            BoostConfig.cmake without a boost_thread component"
        echo "####            -> FindBoost module (CMP0167=OLD)"
    fi
    BOOST_FLAGS+=(-DCMAKE_POLICY_DEFAULT_CMP0167=OLD
                  -DBOOST_ROOT="$BOOST_BASE"
                  -DBoost_NO_BOOST_CMAKE=ON
                  -DBoost_NO_SYSTEM_PATHS=ON)
fi

# -- yaml-cpp ---------------------------------------------------------------
# This is what INSTALL_EXTERNAL_DEPENDENCIES actually covers (yaml-cpp, HepMC);
# it has nothing to do with boost. Prefer the release's own yaml-cpp when scram
# knows about it -- one fewer thing built into the prefix, and no network needed
# -- and leave the option ON so that Hammer self-installs it otherwise.
PREFIX_PATHS=("$BOOST_BASE")
YAMLCPP_BASE=$(scram tool tag yaml-cpp YAML_CPP_BASE 2>/dev/null || true)
if [ -n "$YAMLCPP_BASE" ]; then
    echo "#### yaml-cpp   $YAMLCPP_BASE (from scram)"
    PREFIX_PATHS+=("$YAMLCPP_BASE")
else
    echo "#### yaml-cpp   not a scram tool here -> Hammer will download and build it"
fi
CMAKE_PREFIX_ARG=$(IFS=';'; echo "${PREFIX_PATHS[*]}")

# -- preflight: the bindings' prerequisites, from the RELEASE's interpreter ---
# `cython --version` in the shell says nothing: what matters is what
# .../CMSSW_16_0_8/external/.../bin/python3 can import. Hammer disables
# WITH_PYTHON silently when these are missing, and then builds a C++-only
# library that looks like a success until the first `import hammer`.
echo "#### preflight"
for MOD in Cython numpy; do
    if python3 -c "import $MOD" 2>/dev/null; then
        echo "####   $MOD $(python3 -c "import $MOD; print($MOD.__version__)")"
    else
        echo "####   $MOD NOT importable by $(which python3)" >&2
        echo "[FATAL] the python bindings need $MOD importable by the release's" >&2
        echo "        interpreter, not just a $MOD on PATH. Either add it as a" >&2
        echo "        scram tool or 'pip install --user $MOD' under cmsenv." >&2
        exit 1
    fi
done

mkdir -p "$PREFIX"
if [ ! -d "$SRC/.git" ]; then
    git clone --branch "$TAG" "$REPO" "$SRC"
else
    git -C "$SRC" fetch --all --tags && git -C "$SRC" checkout "$TAG"
fi
echo "#### Hammer source at $(git -C "$SRC" rev-parse --short HEAD)"

# -- source patches, both idempotent -----------------------------------------
# v2.0.0 declares Python >= 3.10 in two places, and CMSSW_16_0_8 ships 3.9.14.
# Both are metadata, not a real requirement: with the floors lowered, the Cython
# bindings compile and link cleanly against 3.9 (Cython 3.1.6, numpy 2.0.2) and
# the resulting pyHammer.so works. Patch here rather than by hand so the build
# stays reproducible -- `git -C $SRC diff` shows exactly what was changed.
PYVER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')

#  (a) the CMake find_package floor: without this WITH_PYTHON is forced OFF and
#      you get a C++-only install that looks like a success.
if grep -q 'find_package(Python3 "3.10" COMPONENTS' "$SRC/CMakeLists.txt"; then
    sed -i "s/find_package(Python3 \"3.10\" COMPONENTS/find_package(Python3 \"$PYVER\" COMPONENTS/" \
        "$SRC/CMakeLists.txt"
    echo "#### patched   CMakeLists.txt: Python3 floor 3.10 -> $PYVER"
fi

#  (b) the wheel's requires-python: the build succeeds and then pip refuses to
#      install its own freshly built wheel.
for META in $(grep -rl -E 'requires-python|python_requires' "$SRC/pyext" 2>/dev/null); do
    if grep -q '3\.10' "$META"; then
        sed -i "s/>=3\.10/>=$PYVER/g; s/>= *3\.10/>= $PYVER/g" "$META"
        echo "#### patched   $(basename "$META"): requires-python -> >=$PYVER"
    fi
done

rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

# -- the flags that matter, and why -----------------------------------------
#  CMAKE_CXX_STANDARD=17 : CMSSW_16 compiles at C++20, where Hammer's aggregate
#                          initialisation no longer compiles. Same workaround as
#                          on the laptop build.
#  WITH_ROOT=OFF         : we never use Hammer's ROOT I/O; linking it against the
#                          release's ROOT only adds ways to fail.
#  PYTHON_USE_CPPYY=OFF  : use the Cython bindings, i.e. the snake_case API
#                          (include_decay, set_ff_eigenvectors, ...). The cppyy
#                          path exposes the CamelCase C++ names instead and none
#                          of our code matches it.
#  INSTALL_EXTERNAL_DEPENDENCIES=ON : yaml-cpp and HepMC ONLY. Hammer's
#                          CMakeLists calls find_package(Boost ... REQUIRED)
#                          unconditionally, so boost always has to come from
#                          outside -- see the resolution block above.
#  the Boost flags       : see the resolution block above.
cmake "$SRC" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DWITH_ROOT=OFF \
    -DWITH_PYTHON=ON \
    -DPYTHON_USE_CPPYY=OFF \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    -DINSTALL_EXTERNAL_DEPENDENCIES=ON \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_ARG" \
    "${BOOST_FLAGS[@]}"

# -- did the bindings survive configure? ------------------------------------
# Hammer forces WITH_PYTHON back to OFF when it cannot build them, so the flag
# we passed is not evidence that they are on. Check the cache, not our intent,
# before spending the compile.
WITH_PYTHON_CACHED=$(cmake -LA -N . 2>/dev/null | sed -n 's/^WITH_PYTHON:BOOL=//p')
if [ "$WITH_PYTHON_CACHED" != "ON" ]; then
    echo "[FATAL] configure left WITH_PYTHON=${WITH_PYTHON_CACHED:-<unset>} despite" >&2
    echo "        -DWITH_PYTHON=ON: Hammer disabled the bindings. Re-run" >&2
    echo "        'cmake . 2>&1 | grep -i -E \"python|cython|numpy\"' in $BUILD" >&2
    echo "        to see why. Building now would produce a C++-only install." >&2
    exit 1
fi
echo "#### bindings   WITH_PYTHON=ON, PYTHON_USE_CPPYY=$(cmake -LA -N . 2>/dev/null | sed -n 's/^PYTHON_USE_CPPYY:BOOL=//p')"

make -j"$JOBS"
make install || true      # the final `pip install` of the wheel may still refuse; handled below

# -- install the wheel into the prefix ---------------------------------------
# The build produces a proper wheel even when the in-tree pip install step
# declines it. Install it explicitly into the prefix, so the module lands
# somewhere the env scripts can point at rather than in a user site-packages.
WHEEL=$(ls -t "$BUILD"/pyext/wrapper_cython/bindings/dist/hammer-*.whl 2>/dev/null | head -1 || true)
if [ -n "$WHEEL" ]; then
    PYSITE="$PREFIX/lib/python$PYVER/site-packages"
    echo "#### wheel      $WHEEL"
    python3 -m pip install --ignore-requires-python --no-deps --upgrade \
        --target "$PYSITE" "$WHEEL"
    echo "#### installed  -> $PYSITE"
else
    echo "[WARN] no wheel found under $BUILD/pyext/wrapper_cython/bindings/dist" >&2
fi

# -- locate what was actually installed -------------------------------------
# Do NOT guess at lib/ vs lib64/ vs site-packages: ask the tree. Hammer's python
# module has moved between versions and a wrong guess looks exactly like a failed
# build ("No module named hammer") hours later, in the wrong shell.
# search the INSTALLED tree only: $PREFIX also holds src/ and build/, and
# src/pyext/wrapper_cython/__init__.py is the template, not an importable module.
SEARCH_DIRS=()
for CAND in "$PREFIX"/lib64 "$PREFIX"/lib "$PREFIX"/python; do
    [ -d "$CAND" ] && SEARCH_DIRS+=("$CAND")
done
MODULE_INIT=''
if [ ${#SEARCH_DIRS[@]} -gt 0 ]; then
    MODULE_INIT=$(find "${SEARCH_DIRS[@]}" -name '__init__.py' -path '*hammer*' -print -quit 2>/dev/null || true)
    if [ -z "$MODULE_INIT" ]; then
        MODULE_INIT=$(find "${SEARCH_DIRS[@]}" \( -name 'hammer*.so' -o -name 'Hammer*.so' \) -print -quit 2>/dev/null || true)
    fi
fi
if [ -z "$MODULE_INIT" ]; then
    echo "[FATAL] installed, but no python module found under $PREFIX." >&2
    echo "        The C++ library built and the bindings did not. Check the" >&2
    echo "        configure summary:  cmake -LA $BUILD | grep -i -E 'python|cppyy'" >&2
    exit 1
fi
PYDIR=$(dirname "$(dirname "$MODULE_INIT")")
if [ "$(basename "$MODULE_INIT")" != '__init__.py' ]; then
    PYDIR=$(dirname "$MODULE_INIT")
fi
echo "#### python module $MODULE_INIT"
echo "####   PYTHONPATH  $PYDIR"

# -printf is GNU-only; -exec dirname keeps this working on the Mac too.
LIBDIRS=$(find "$PREFIX"/lib64 "$PREFIX"/lib -name 'libHammer*' -exec dirname {} \; 2>/dev/null \
          | sort -u | tr '\n' ':' | sed 's/:$//')
LIBDIRS=${LIBDIRS:-$PREFIX/lib}
echo "####   LD_LIBRARY  $LIBDIRS"

# -- env scripts, both shells, with the resolved paths baked in --------------
cat > "$PREFIX/hammer_env.sh" <<EOF
# generated by install_hammer.sh on $(date -Is) -- bash
export HAMMER_PREFIX=$PREFIX
export PYTHONPATH=$PYDIR\${PYTHONPATH:+:\$PYTHONPATH}
export LD_LIBRARY_PATH=$LIBDIRS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF

cat > "$PREFIX/hammer_env.csh" <<EOF
# generated by install_hammer.sh on $(date -Is) -- tcsh
setenv HAMMER_PREFIX $PREFIX
if (! \$?PYTHONPATH) setenv PYTHONPATH ""
if (! \$?LD_LIBRARY_PATH) setenv LD_LIBRARY_PATH ""
if ("\$PYTHONPATH" == "") then
    setenv PYTHONPATH $PYDIR
else
    setenv PYTHONPATH $PYDIR:\${PYTHONPATH}
endif
if ("\$LD_LIBRARY_PATH" == "") then
    setenv LD_LIBRARY_PATH $LIBDIRS
else
    setenv LD_LIBRARY_PATH ${LIBDIRS}:\${LD_LIBRARY_PATH}
endif
EOF

echo
echo "#### installed. Now:"
echo "####   source $PREFIX/hammer_env.sh        # bash / SLURM job scripts"
echo "####   source $PREFIX/hammer_env.csh       # tcsh: the login shell on t3ui"
echo "####   python3 \$CMSSW_BASE/src/Bmmm/Analysis/test/rjpsi/hammer/check_hammer_api.py"
echo "#### which asserts the Cython (snake_case) API and every method HammerFF calls."
