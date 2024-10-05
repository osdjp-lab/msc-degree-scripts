#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Get file listing for given set of package names

# Verify number of positional parameters

if [ $# -eq 3 ]; then
    INPUT_LST="$(realpath "$1")"
    PKG_LISTING_DIR="$(realpath "$2")"
    OUTPUT_LST="$(realpath "$3")"
else
    printf "Usage: %s [INPUT_LST] [PKG_LISTING_DIR] [OUTPUT_LST]\n" "$(basename "$0")"
    exit
fi

# Verify values of INPUT_LST and PKG_LISTING_DIR

if ! [ -r "$INPUT_LST" ]; then
    printf "INPUT_LST file is not readable or does not exist\n"
    exit
fi

if ! [ -d "$PKG_LISTING_DIR" ]; then
    printf "PKG_LISTING_DIR does not exist or is not a directory\n"
    exit
fi

if [ -r "$OUTPUT_LST" ]; then
    printf "OUTPUT_LST file exists\n"
    exit
fi

########################################

while IFS= read -r pkgname; do
    FOUND_FILES=$(find "$PKG_LISTING_DIR" -type f -name "*$pkgname*" -exec basename -- {} \;)
    if [ "$(echo "$FOUND_FILES" | wc -l)" -ne 1 ]; then
        FOUND_FILES=$(echo "$FOUND_FILES" | fzf --query "$pkgname")
    fi
    LST_FILE=$(find "$PKG_LISTING_DIR" -type f -name "$FOUND_FILES")
    cat "$LST_FILE" >> "$OUTPUT_LST"
done < "$INPUT_LST"

sort "$OUTPUT_LST" -o "$OUTPUT_LST"

