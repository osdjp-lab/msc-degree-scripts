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
    printf "Destination \"%s\" does not exist or is not a directory\n" "$PKG_META_DIR"
    exit
fi

if ! [ -d "$NAME_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$NAME_DIR"
    exit
fi

########################################

ls "$PKG_META_DIR" > "$NAME_DIR/base"

ex -s "$NAME_DIR/base" << EOF
set verbose=1
%s/\.txt//g
w
q
EOF

cp "$NAME_DIR/base" "$NAME_DIR/simplified"

ex -s "$NAME_DIR/simplified" << EOF
set verbose=1
%s/FreeBSD-//g
w
q
EOF

sort "$NAME_DIR/simplified" -o "$NAME_DIR/simplified"

