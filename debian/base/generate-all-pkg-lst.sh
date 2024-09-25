#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate all package contents listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    CONTENTS_LISTING="$(realpath "$1")"
    NAME_DIR="$(realpath "$2")"
else
    printf "Usage: %s [CONTENTS_LISTING] [NAME_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify CONTENTS_LISTING and NAME_DIR

if ! [ -r "$CONTENTS_LISTING" ]; then
    printf "File CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

if ! [ -d "$NAME_DIR" ]; then
    printf "Destination NAME_DIR does not exist or is not a directory\n"
    exit
fi

########################################

# All package file listing

ALL_PKG_LST="$NAME_DIR/all-pkg.lst"

cp -v "$CONTENTS_LISTING" "$ALL_PKG_LST"

# Remove name column and empty spaces at ends of lines

ex -s "$ALL_PKG_LST" << EOF
%s/^\(.*\)\s.*$/\1/g
%s/\s*$//g
w
q
EOF

sort "$ALL_PKG_LST" -o "$ALL_PKG_LST"

uniq "$ALL_PKG_LST" | sponge "$ALL_PKG_LST"

