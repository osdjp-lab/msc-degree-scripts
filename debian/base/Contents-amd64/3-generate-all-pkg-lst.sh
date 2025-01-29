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
    ALL_PKG_LISTING="$(realpath "$2")"
else
    printf "Usage: %s [CONTENTS_LISTING] [ALL_PKG_LISTING]\n" "$(basename "$0")"
    exit
fi

# Verify CONTENTS_LISTING

if ! [ -r "$CONTENTS_LISTING" ]; then
    printf "File CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

########################################

# Copy contents listing file to all package listing file

cp -v "$CONTENTS_LISTING" "$ALL_PKG_LISTING"

# Remove name column and empty spaces at ends of lines

ex -s "$ALL_PKG_LISTING" << EOF
%s/^\(.*\)\s.*$/\1/g
%s/\s*$//g
w
q
EOF

sort "$ALL_PKG_LISTING" -o "$ALL_PKG_LISTING"

# uniq "$ALL_PKG_LISTING" | sponge "$ALL_PKG_LISTING"

