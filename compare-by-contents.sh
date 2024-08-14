#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Compare package file contents listings from FreeBSD base
# and Debian bookworm main 

# Results are split into two files
# - common (existing in both system listings)
# - freebsd-only (existing only in FreeBSD package content listing)
# - debian-only (existing only in Debian package content listing)

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

# Check if "all-pkg.lst" file is present and readable in SRC_DIR subdirectories

DEBIAN_LST="$SRC_DIR/debian/all-pkg.lst"
FREEBSD_LST="$SRC_DIR/freebsd/base/all-pkg.lst"

if ! [ -r "$DEBIAN_LST" ]; then
    printf "File \"debian/all-pkg.lst\" in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -r "$FREEBSD_LST" ]; then
    printf "File \"freebsd/base/all-pkg.lst\" in source directory is not readable or does not exist\n"
    exit
fi

# Generate results

comm -12 "$FREEBSD_LST" "$DEBIAN_LST" > "$DEST_DIR/common"
comm -23 "$FREEBSD_LST" "$DEBIAN_LST" > "$DEST_DIR/freebsd-only"
comm -13 "$FREEBSD_LST" "$DEBIAN_LST" > "$DEST_DIR/debian-only"

