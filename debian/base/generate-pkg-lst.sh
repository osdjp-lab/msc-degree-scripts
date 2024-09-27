#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract package file listings

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    NAME_DIR="$(realpath "$1")"
    LISTING_DIR="$(realpath "$2")"
else
    printf "Usage: %s [NAME_DIR] [LISTING_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify NAME_DIR and LISTING_DIR

if ! [ -d "$NAME_DIR" ]; then
    printf "Destination NAME_DIR does not exist or is not a directory\n"
    exit
fi

if ! [ -d "$LISTING_DIR" ]; then
    printf "Output listing files directory \"%s\" does not exist or is not a directory\n" "$LISTING_DIR"
    exit
fi

########################################

PKG_CONTENTS_LISTING="$NAME_DIR/pkg-contents"

cd "$LISTING_DIR" || exit

awk '
{
    n = split($1, lines, "/")
    filename = lines[n]
    print $2 >> filename ".txt"
}' "$PKG_CONTENTS_LISTING"

cd - || exit

