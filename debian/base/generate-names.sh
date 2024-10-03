#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package name and contents listings

# Verify number of positional parameters

if [ $# -eq 3 ]; then
    CONTENTS_LISTING="$(realpath "$1")"
    PKG_META_DIR="$(realpath "$2")"
    NAME_DIR="$(realpath "$3")"
else
    printf "Usage: %s [CONTENTS_LISTING] [PKG_META_DIR] [NAME_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify CONTENTS_LISTING PKG_META_DIR and NAME_DIR

if ! [ -r "$CONTENTS_LISTING" ]; then
    printf "File CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

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

# Extract package-contents listing

PKG_CONTENTS_LISTING="$NAME_DIR/pkg-contents"

cp -v "$CONTENTS_LISTING" "$PKG_CONTENTS_LISTING"

# Switch columns and replace separator with '\t'

ex -s "$PKG_CONTENTS_LISTING" << EOF
%s/^\(.*\)\s\(.*\)$/\2\t\1/g
w
q
EOF

cut -d "/" -f "2-" "$PKG_CONTENTS_LISTING" | sponge "$PKG_CONTENTS_LISTING"

sort "$PKG_CONTENTS_LISTING" -o "$PKG_CONTENTS_LISTING"

# Make list of deduplicated package names

DEDUPLICATED_PKGS_LISTING="$NAME_DIR/deduplicated-pkgs"

cut -f1 "$PKG_CONTENTS_LISTING" > "$DEDUPLICATED_PKGS_LISTING"

sort "$DEDUPLICATED_PKGS_LISTING" -o "$DEDUPLICATED_PKGS_LISTING"

uniq "$DEDUPLICATED_PKGS_LISTING" | sponge "$DEDUPLICATED_PKGS_LISTING"

