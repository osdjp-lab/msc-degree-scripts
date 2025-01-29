#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Download FreeBSD:14:amd64 "packagesite.yaml" metadata file for FreeBSD:14:amd64/release_1

BASE_URL="http://pkg.freebsd.org/FreeBSD:14:amd64"
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

PACKAGESITE_XZ="${DEST_DIR}/packagesite.txz"

wget "$RELEASE_1_SRC" -O "$PACKAGESITE_XZ"

unxz "$PACKAGESITE_XZ"

PACKAGESITE_TAR="${DEST_DIR}/packagesite.tar"

tar -xf "$PACKAGESITE_TAR" -C "${DEST_DIR}" "packagesite.yaml"

rm "$PACKAGESITE_TAR"

