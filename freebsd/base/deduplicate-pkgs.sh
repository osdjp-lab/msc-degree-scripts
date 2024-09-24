#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract deduplicated packages into separate directory

# Verify number of positional parameters

if [ $# -eq 3 ]; then
    NAME_DIR="$(realpath "$1")"
    PKG_META_DIR="$(realpath "$2")"
    DEDUPLICATED_PKGS="$(realpath "$3")"
else
    printf "Usage: %s [NAME_DIR] [PKG_META_DIR] [DEDUPLICATED_PKGS]\n" "$(basename "$0")"
    exit
fi

# Verify NAME_DIR and DEDUPLICATED_PKGS

if ! [ -d "$NAME_DIR" ]; then
    printf "Name directory \"%s\" does not exist or is not a directory\n" "$NAME_DIR"
    exit
fi

if ! [ -d "$PKG_META_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$PKG_META_DIR"
    exit
fi

if ! [ -d "$DEDUPLICATED_PKGS" ]; then
    printf "Deduplicated pkgs directory \"%s\" does not exist or is not a directory\n" "$DEDUPLICATED_PKGS"
    exit
fi

########################################

DEDUPLICATED_LISTING="$NAME_DIR/no-man-dbg-dev-lib32"

while IFS= read -r pkgname; do
    ln -vsr "$(find "$PKG_META_DIR" -type f -name "FreeBSD-$pkgname.txt")" "$DEDUPLICATED_PKGS"
done < "$DEDUPLICATED_LISTING"

