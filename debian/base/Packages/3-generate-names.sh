#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package name listings

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_META_DIR="$(realpath "$1")"
    NAME_DIR="$(realpath "$2")"
else
    printf "Usage: %s [PKG_META_DIR] [NAME_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_META_DIR and NAME_DIR

if ! [ -d "$PKG_META_DIR" ]; then
    printf "Destination PKG_META_DIR does not exist or is not a directory\n"
    exit
fi

if ! [ -d "$NAME_DIR" ]; then
    printf "Destination NAME_DIR does not exist or is not a directory\n"
    exit
fi

########################################

# Generate package listing

PKG_LISTING="$NAME_DIR/pkgs"

ls "$PKG_META_DIR" > "$PKG_LISTING"

sort "$PKG_LISTING" -o "$PKG_LISTING"

ex -s "$PKG_LISTING" << EOF
%s/\.txt$//g
w
q
EOF

