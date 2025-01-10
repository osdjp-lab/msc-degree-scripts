#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Find Debian packages complementary to FreeBSD base packages

# Verify number of positional parameters

if [ $# -eq 5 ]; then
    FREEBSD_PKG_LST_DIR="$(realpath "$1")"
    DEBIAN_ALL_PKG_LST="$(realpath "$2")"
    DEBIAN_PKG_CONTENTS_LST="$(realpath "$3")"
    DEBIAN_PKG_LST_DIR="$(realpath "$4")"
    OUTPUT_DIR="$(realpath "$5")"
else
    printf "Usage: %s [FREEBSD_PKG_LST_DIR] [DEBIAN_ALL_PKG_LST] [DEBIAN_PKG_CONTENTS_LST] [DEBIAN_PKG_LST_DIR] [OUTPUT_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify arguments

if ! [ -d "$FREEBSD_PKG_LST_DIR" ]; then
    printf "Source FREEBSD_PKG_LST_DIR does not exist or is not a directory\n"
    exit
fi

if ! [ -r "$DEBIAN_ALL_PKG_LST" ]; then
    printf "File DEBIAN_ALL_PKG_LST in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -r "$DEBIAN_PKG_CONTENTS_LST" ]; then
    printf "File DEBIAN_PKG_CONTENTS_LST in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -d "$DEBIAN_PKG_LST_DIR" ]; then
    printf "Destination DEBIAN_PKG_LST_DIR does not exist or is not a directory\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR"

for freebsd_pkg in "$FREEBSD_PKG_LST_DIR"/*; do
    pkg_name="$(basename "$freebsd_pkg" | sed -e "s/FreeBSD-\(.*\)-.*/\1/")"
    TARGET_DIR="$OUTPUT_DIR/$pkg_name"
    mkdir -pv "$TARGET_DIR"
    comm -12 "$freebsd_pkg" "$DEBIAN_ALL_PKG_LST" > "$TARGET_DIR/common"
    comm -23 "$freebsd_pkg" "$DEBIAN_ALL_PKG_LST" > "$TARGET_DIR/freebsd-only"
    comm -13 "$freebsd_pkg" "$DEBIAN_ALL_PKG_LST" > "$TARGET_DIR/debian-only"
    "$SCRIPT_DIR/names-from-lst.sh" "$TARGET_DIR/common" "$DEBIAN_PKG_CONTENTS_LST" "$TARGET_DIR/debian-matched-pkgs"
    while IFS= read -r debian_pkg; do
        pkg_lst="$DEBIAN_PKG_LST_DIR/$debian_pkg.txt"
        pkg_dir="$TARGET_DIR/$debian_pkg"
        mkdir -pv "$pkg_dir"
        comm -12 "$TARGET_DIR/common" "$pkg_lst" > "$pkg_dir/matched"
        comm -13 "$TARGET_DIR/common" "$pkg_lst" > "$pkg_dir/unmatched"
    done < "$TARGET_DIR/debian-matched-pkgs"
done

