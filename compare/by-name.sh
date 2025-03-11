#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Compare packages from FreeBSD base and Debian bookworm main
# on the basis of package names

# Verify number of positional parameters

if [ $# -eq 3 ]; then
    DEBIAN_PKG_NAMES="$(realpath "$1")"
    FREEBSD_PKG_NAMES="$(realpath "$2")"
    DEST_DIR="$(realpath "$3")"
else
    printf "Usage: %s [DEBIAN_PKG_NAMES] [FREEBSD_PKG_NAMES] [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify values of DEBIAN_PKG_NAMES, FREEBSD_PKG_NAMES and DEST_DIR

if ! [ -r "$DEBIAN_PKG_NAMES" ]; then
    printf "File \"debian/pkg-names\" in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -r "$FREEBSD_PKG_NAMES" ]; then
    printf "File \"freebsd/base/names/simplified\" in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$DEST_DIR"
    exit
fi

# Generate initial exact match comparison

comm -12 "$FREEBSD_PKG_NAMES" "$DEBIAN_PKG_NAMES" > "$DEST_DIR/common"
comm -23 "$FREEBSD_PKG_NAMES" "$DEBIAN_PKG_NAMES" > "$DEST_DIR/freebsd-only"
comm -13 "$FREEBSD_PKG_NAMES" "$DEBIAN_PKG_NAMES" > "$DEST_DIR/debian-only"

# Generate partial matches

# (...)

