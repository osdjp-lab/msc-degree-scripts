#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Download debian-bookworm main package metadata
# files "Packages" and "Contents-amd64"

BASE_URL="https://ftp.debian.org/debian/dists/bookworm/main"
PACKAGES_SRC="${BASE_URL}/binary-amd64/Packages.gz"
CONTENTS_SRC="${BASE_URL}/Contents-amd64.gz"

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

PACKAGES_GZ="${DEST_DIR}/Packages.gz"
CONTENTS_GZ="${DEST_DIR}/Contents-amd64.gz"

wget "$PACKAGES_SRC" -O "$PACKAGES_GZ"
wget "$CONTENTS_SRC" -O "$CONTENTS_GZ"

gunzip "$PACKAGES_GZ"
gunzip "$CONTENTS_GZ"

