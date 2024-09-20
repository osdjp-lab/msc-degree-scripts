#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package name listings

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    NAME_DIR="$(realpath "$1")"
    PKG_META_DIR="$(realpath "$2")"
    MAN_DIR="$(realpath "$3")"
else
    printf "Usage: %s [NAME_DIR] [PKG_META_DIR] [MAN_DIR]\n" "$(basename "$0")"
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

if ! [ -d "$MAN_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$MAN_DIR"
    exit
fi

########################################

BASE_LISTING="$NAME_DIR/base"

ls "$PKG_META_DIR" > "$BASE_LISTING"

ex -s "$BASE_LISTING" << EOF
set verbose=1
%s/\.txt//g
w
q
EOF

SIMPLIFIED_LISTING="$NAME_DIR/simplified"

cp "$BASE_LISTING" "$SIMPLIFIED_LISTING"

ex -s "$SIMPLIFIED_LISTING" << EOF
set verbose=1
%s/FreeBSD-//g
w
q
EOF

sort "$SIMPLIFIED_LISTING" -o "$SIMPLIFIED_LISTING"

# Create deduplicated package name listing

DEDUPLICATED_LISTING="$WORK_DIR/names/no-man-dbg-dev-lib32"

cp -v "$SIMPLIFIED_LISTING" "$DEDUPLICATED_LISTING"

ex -s "$DEDUPLICATED_LISTING" << EOF
set verbose=1
g/-dbg$/d
g/-dbg-/d
g/-man$/d
g/-dev$/d
g/-dev-/d
g/-lib32$/d
w
q
EOF

# Generate listing of packages without man pages

MAN_PKGS="$WORK_DIR/names/man-pkgs"
NO_MAN_PKGS="$WORK_DIR/names/no-man-pkgs"

ls "$MAN_DIR" > "$MAN_PKGS"

sort "$MAN_PKGS" -o "$MAN_PKGS"

comm -23 "$DEDUPLICATED_LISTING" "$MAN_PKGS" > "$NO_MAN_PKGS"

