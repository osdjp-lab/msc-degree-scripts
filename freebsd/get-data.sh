#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Download FreeBSD:14:amd64 "packagesite.yaml" metadata files and individual package files for FreeBSD:14:amd64/base_release_1

BASE_URL="http://pkg.freebsd.org/FreeBSD:14:amd64"
BASE_PKGS="$BASE_URL/base_release_1/"
BASE_RELEASE_1_SRC="$BASE_URL/base_release_1/packagesite.txz"
RELEASE_1_SRC="$BASE_URL/release_1/packagesite.txz"

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

# Get base_release_1 data

# Get packagesite.yaml

BASE_DIR="$DEST_DIR/base"

mkdir "$BASE_DIR"

PACKAGESITE_XZ="${BASE_DIR}/packagesite.txz"

wget "$BASE_RELEASE_1_SRC" -O "$PACKAGESITE_XZ"

unxz "$PACKAGESITE_XZ"

PACKAGESITE_TAR="${BASE_DIR}/packagesite.tar"

tar -xf "$PACKAGESITE_TAR" -C "${BASE_DIR}" "packagesite.yaml"

rm "$PACKAGESITE_TAR"

# Get package files

PKG_LIST="${BASE_DIR}/pkg-urls"

wget "$BASE_PKGS" -O "$PKG_LIST"

ex -s "$PKG_LIST" << EOF
%s/^.*\(FreeBSD.*\.pkg\).*$/\1/g
g!/FreeBSD.*\.pkg/d
%s?^?$BASE_PKGS?g
w
q
EOF

PKG_DIR="${BASE_DIR}/pkgs"

mkdir "$PKG_DIR"

wget -i "$PKG_LIST" -P "$PKG_DIR"

rm "$PKG_LIST"

# Get release_1 data

RELEASE_DIR="$DEST_DIR/release"

mkdir "$RELEASE_DIR"

PACKAGESITE_XZ="${RELEASE_DIR}/packagesite.txz"

wget "$RELEASE_1_SRC" -O "$PACKAGESITE_XZ"

unxz "$PACKAGESITE_XZ"

PACKAGESITE_TAR="${RELEASE_DIR}/packagesite.tar"

tar -xf "$PACKAGESITE_TAR" -C "${RELEASE_DIR}" "packagesite.yaml"

rm "$PACKAGESITE_TAR"


