#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract package man pages

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_FILES_DIR="$(realpath "$1")"
    MAN_DIR="$(realpath "$2")"
else
    printf "Usage: %s [PKG_FILES_DIR] [MAN_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_FILES_DIR and MAN_DIR

if ! [ -d "$PKG_FILES_DIR" ]; then
    printf "Package files directory \"%s\" does not exist or is not a directory\n" "$PKG_FILES_DIR"
    exit
fi

if ! [ -d "$MAN_DIR" ]; then
    printf "Man directory \"%s\" does not exist or is not a directory\n" "$PKG_FILES_DIR"
    exit
fi

########################################

TMP_DIR="$(mktemp -d)"
mkdir "$TMP_DIR"

MAN_PKGS_LST="$(mktemp)"

find "$PKG_FILES_DIR" -name "*man*" > "$MAN_PKGS_LST"

cd "$TMP_DIR" || exit

while IFS= read -r pkg; do
    printf "%s\n" "$pkg"
    bsdtar -xvf "$pkg"
    rm "+MANIFEST" "+COMPACT_MANIFEST"
    pkg_name="$(basename "$pkg" | sed "s/FreeBSD-\(.*\)-man.*/\1/")"
    mkdir -pv "$MAN_DIR/$pkg_name"
    find . -type f -exec mv -vt "$MAN_DIR/$pkg_name" {} +
done < "$MAN_PKGS_LST"

cd - || exit

rm "$MAN_PKGS_LST"

find "$TMP_DIR" -type d -empty -delete

