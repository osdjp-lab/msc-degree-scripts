#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Get package names for a given file listing

# Verify number of positional parameters

if [ $# -eq 3 ]; then
    INPUT_LST="$(realpath "$1")"
    PKG_CONTENTS_LISTING="$(realpath "$2")"
    OUTPUT_LST="$(realpath "$3")"
else
    printf "Usage: %s [INPUT_LST] [PKG_CONTENTS_LISTING] [OUTPUT_LST]\n" "$(basename "$0")"
    exit
fi

# Verify values of INPUT_LST and PKG_CONTENTS_LISTING

if ! [ -r "$INPUT_LST" ]; then
    printf "INPUT_LST file is not readable or does not exist\n"
    exit
fi

if ! [ -r "$PKG_CONTENTS_LISTING" ]; then
    printf "PKG_CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

if [ -r "$OUTPUT_LST" ]; then
    printf "OUTPUT_LST file exists\n"
    exit
fi

########################################

while IFS= read -r listing; do
    printf "%s\n" "$listing"
    grep -F "$listing" "$PKG_CONTENTS_LISTING" | cut -f1 >> "$OUTPUT_LST"
done < "$INPUT_LST"

sort "$OUTPUT_LST" -o "$OUTPUT_LST"

uniq "$OUTPUT_LST" | sponge "$OUTPUT_LST"

