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

DEDUPLICATED_PKGS_LISTING="$NAME_DIR/deduplicated-pkgs"
PKG_CONTENTS_LISTING="$NAME_DIR/pkg-contents"

# Create empty output files

while IFS= read -r pkg
do
    true > "$LISTING_DIR/$pkg.lst"
done < "$DEDUPLICATED_PKGS_LISTING"

# Sort contents of original listing into previously created files

while IFS= read -r line
do
    pkg_name=$(printf "%s" "$line" | cut -f1 | rev | cut -d'/' -f1 | rev)
    printf "%s\n" "$pkg_name"
    printf "%s\n" "$line" | cut -f2 >> "$LISTING_DIR/$pkg_name.lst"
    sort "$LISTING_DIR/$pkg_name.lst" -o "$LISTING_DIR/$pkg_name.lst"
done < "$PKG_CONTENTS_LISTING"

