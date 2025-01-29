#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package-contents listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SIMPLIFIED_LISTING_DIR="$(realpath "$1")"
    NAME_DIR="$(realpath "$2")"
else
    printf "Usage: %s [SIMPLIFIED_LISTING_DIR] [NAME_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify SIMPLIFIED_LISTING_DIR and NAME_DIR

if ! [ -d "$SIMPLIFIED_LISTING_DIR" ]; then
    printf "Input listing directory \"%s\" does not exist or is not a directory\n" "$SIMPLIFIED_LISTING_DIR"
    exit
fi

if ! [ -d "$NAME_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$NAME_DIR"
    exit
fi

########################################

PKG_CONTENTS_LISTING="$NAME_DIR/pkg-contents"

for pkg_lst in "$SIMPLIFIED_LISTING_DIR"/*; do
    tmp_lst="$(mktemp)"
    pkg_name="$(basename "$pkg_lst" | sed -e "s/FreeBSD-\(.*\)-.*/\1/")"
    cp "$pkg_lst" "$tmp_lst"
    ex -s "$tmp_lst" << EOF
set verbose=1
%s/^/$pkg_name\t/g
w
q
EOF
    cat "$tmp_lst" >> "$PKG_CONTENTS_LISTING"
    rm "$tmp_lst"
done

sort "$PKG_CONTENTS_LISTING" -o "$PKG_CONTENTS_LISTING"

