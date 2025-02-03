#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate package name and contents listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    CONTENTS_LISTING="$(realpath "$1")"
    PKG_CONTENTS_LISTING="$(realpath "$2")"
else
    printf "Usage: %s [CONTENTS_LISTING] [PKG_CONTENTS_LISTING]\n" "$(basename "$0")"
    exit
fi

# Verify CONTENTS_LISTING

if ! [ -r "$CONTENTS_LISTING" ]; then
    printf "File CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"

# Switch columns

awk -f "$WORK_DIR/switch-columns.awk" "$CONTENTS_LISTING" \
    > "$PKG_CONTENTS_LISTING"

# Split comma separated entries into multiple entries

awk -f "$WORK_DIR/split-multi-pkg-entries.awk" \
    "$PKG_CONTENTS_LISTING" | sponge "$PKG_CONTENTS_LISTING"

sort "$PKG_CONTENTS_LISTING" -o "$PKG_CONTENTS_LISTING"

