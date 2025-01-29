#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Download "packagesite.yaml" metadata file and individual package files
# for FreeBSD:14:amd64/base_release_1

BASE_URL="http://pkg.freebsd.org/FreeBSD:14:amd64"
BASE_PKGS="$BASE_URL/base_release_1/"
BASE_RELEASE_1_SRC="$BASE_URL/base_release_1/packagesite.txz"

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    DEST_DIR="$(realpath "$1")"
else
    printf "Usage: %s [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify DEST_DIR

if ! [ -d "$DEST_DIR" ]; then
    printf "DEST_DIR does not exist or is not a directory\n"
    exit
fi

########################################

# Get packagesite.yaml

PACKAGESITE_XZ="${DEST_DIR}/packagesite.txz"

wget "$BASE_RELEASE_1_SRC" -O "$PACKAGESITE_XZ"

unxz "$PACKAGESITE_XZ"

PACKAGESITE_TAR="${DEST_DIR}/packagesite.tar"

tar -xf "$PACKAGESITE_TAR" -C "${DEST_DIR}" "packagesite.yaml"

rm "$PACKAGESITE_TAR"

# Get package files

PKG_LIST="${DEST_DIR}/pkg-urls"

wget "$BASE_PKGS" -O "$PKG_LIST"

ex -s "$PKG_LIST" << EOF
%s/^.*\(FreeBSD.*\.pkg\).*$/\1/g
g!/FreeBSD.*\.pkg/d
%s?^?$BASE_PKGS?g
w
q
EOF

PKG_DIR="${DEST_DIR}/pkgs"

mkdir "$PKG_DIR"

wget -i "$PKG_LIST" -P "$PKG_DIR"

rm "$PKG_LIST"

