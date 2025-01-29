#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package name and category listings

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    NAME_DIR="$(realpath "$1")"
    PKG_META_DIR="$(realpath "$2")"
else
    printf "Usage: %s [NAME_DIR] [PKG_META_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify NAME_DIR, PKG_META_DIR and MAN_DIR

if ! [ -d "$NAME_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$NAME_DIR"
    exit
fi

if ! [ -d "$PKG_META_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$PKG_META_DIR"
    exit
fi

########################################

# Generate package name listing

NAME_LISTING="$NAME_DIR/names"

ls "$PKG_META_DIR" > "$NAME_LISTING"

ex -s "$NAME_LISTING" << EOF
set verbose=1
%s/\.txt//g
w
q
EOF

# Generate category listing

CATEGORY_LISTING="$NAME_DIR/categories"

grep -ri '"categories"' "$PKG_META_DIR" > "$CATEGORY_LISTING"

ex -s "$CATEGORY_LISTING" << EOF
set verbose=1
%s/^.*categories":\[//g
%s/\]//g
%s/,/\r/g
%s/"//g
w
q
EOF

sort -u "$CATEGORY_LISTING" -o "$CATEGORY_LISTING"

