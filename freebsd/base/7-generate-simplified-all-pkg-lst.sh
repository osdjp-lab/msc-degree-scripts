#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Create deduplicated single file all base system package file listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SIMPLIFIED_LISTING_DIR="$(realpath "$1")"
    ALL_PKG_LST="$(realpath "$2")"
else
    printf "Usage: %s [SIMPLIFIED_LISTING_DIR] [ALL_PKG_LST]\n" "$(basename "$0")"
    exit
fi

# Verify SIMPLIFIED_LISTING_DIR and ALL_PKG_LST

if ! [ -d "$SIMPLIFIED_LISTING_DIR" ]; then
    printf "Input listing files directory \"%s\" does not exist or is not a directory\n" "$SIMPLIFIED_LISTING_DIR"
    exit
fi

if [ -f "$ALL_PKG_LST" ]; then
    printf "Output all package listing file \"%s\" exists\n" "$ALL_PKG_LST"
    exit
fi

########################################

cat "$SIMPLIFIED_LISTING_DIR"/* > "$ALL_PKG_LST"

sort "$ALL_PKG_LST" -o "$ALL_PKG_LST"

uniq "$ALL_PKG_LST" | sponge "$ALL_PKG_LST"

