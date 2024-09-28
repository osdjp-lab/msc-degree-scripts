#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Compare packages from FreeBSD base and Debian bookworm main
# on the basis of package names

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SRC_DIR="$(realpath "$1")"
    DEST_DIR="$(realpath "$2")"
else
    printf "Usage: %s [SRC_DIR] [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify values of SRC_DIR and DEST_DIR

if ! [ -d "$SRC_DIR" ]; then
    printf "Source \"%s\" does not exist or is not a directory\n" "$SRC_DIR"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$DEST_DIR"
    exit
fi

# Check if package name list files are present and readable
# in SRC_DIR subdirectories

DEBIAN_PKG_NAMES="$SRC_DIR/debian/names/deduplicated-pkgs"
FREEBSD_PKG_NAMES="$SRC_DIR/freebsd/base/names/no-man-dbg-dev-lib32"

if ! [ -r "$DEBIAN_PKG_NAMES" ]; then
    printf "File \"debian/pkg-names\" in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -r "$FREEBSD_PKG_NAMES" ]; then
    printf "File \"freebsd/base/names/simplified\" in source directory is not readable or does not exist\n"
    exit
fi

# Generate initial comparison results

TARGET_DIR="$DEST_DIR/by-name"
mkdir -pv "$TARGET_DIR"

comm -12 "$FREEBSD_PKG_NAMES" "$DEBIAN_PKG_NAMES" > "$TARGET_DIR/common"
comm -23 "$FREEBSD_PKG_NAMES" "$DEBIAN_PKG_NAMES" > "$TARGET_DIR/freebsd-only"
comm -13 "$FREEBSD_PKG_NAMES" "$DEBIAN_PKG_NAMES" > "$TARGET_DIR/debian-only"

