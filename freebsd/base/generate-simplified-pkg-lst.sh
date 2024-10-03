#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract package file listings

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    LISTING_DIR="$(realpath "$1")"
    SIMPLIFIED_LISTING_DIR="$(realpath "$2")"
else
    printf "Usage: %s [LISTING_DIR] [SIMPLIFIED_LISTING_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify LISTING_DIR and SIMPLIFIED_LISTING_DIR 

if ! [ -d "$LISTING_DIR" ]; then
    printf "Output listing directory \"%s\" does not exist or is not a directory\n" "$LISTING_DIR"
    exit
fi

if ! [ -d "$SIMPLIFIED_LISTING_DIR" ]; then
    printf "Input listing directory \"%s\" does not exist or is not a directory\n" "$SIMPLIFIED_LISTING_DIR"
    exit
fi

########################################

cp -Rv "$LISTING_DIR"/. "$SIMPLIFIED_LISTING_DIR"

for pkg in "$SIMPLIFIED_LISTING_DIR"/*; do
    ex -s "$pkg" << EOF
g/"files":/d
g/"config":/d
g/"directories":/d
g/^$/d
%s/:"y"//g
%s/"//g
%s/^\///g
w
q
EOF
    sort "$pkg" -o "$pkg"
done

