#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package name listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_CONTENTS_LISTING="$(realpath "$1")"
    PKG_LISTING="$(realpath "$2")"
else
    printf "Usage: %s [PKG_CONTENTS_LISTING] [PKG_LISTING]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_CONTENTS_LISTING

if ! [ -r "$PKG_CONTENTS_LISTING" ]; then
    printf "File PKG_CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

########################################

# Make list of deduplicated package names

cut -d' ' -f1 "$PKG_CONTENTS_LISTING" > "$PKG_LISTING"

sort "$PKG_LISTING" -o "$PKG_LISTING"

uniq "$PKG_LISTING" | sponge "$PKG_LISTING"

