#!/bin/bash -e

# install location
prefix="${1:-${HOME}/local}"

# wxPython
url="http://downloads.sourceforge.net/wxpython/wxPython-src-2.8.10.1.tar.bz2"
dir=$( basename "${url}" .tar.bz2 )

cd "${prefix}"
curl "${url}" | tar jx
cd "${dir}"
mkdir bld
cd bld

../configure \
    --prefix="${prefix}" \
    --enable-optimize \
    --enable-debug_flag \
    --enable-monolithic

make -C contrib/src/gizmos install
make -C contrib/src/stc install

cd ../wxPython
python setup.py build_ext --inplace
python setup.py install

export LD_LIBRARY_PATH="${prefix}/lib"
