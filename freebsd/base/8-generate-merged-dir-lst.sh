#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate merged listings of directories of interest

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SPLIT_LISTINGS_DIR="$(realpath "$1")"
    MERGED_LISTINGS_DIR="$(realpath "$2")"
else
    printf "Usage: %s [SPLIT_LISTINGS_DIR] [MERGED_LISTINGS_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify SPLIT_LISTINGS_DIR and MERGED_LISTINGS_DIR 

if ! [ -d "$SPLIT_LISTINGS_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$SPLIT_LISTINGS_DIR"
    exit
fi

if ! [ -d "$MERGED_LISTINGS_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$MERGED_LISTINGS_DIR"
    exit
fi

########################################

# Binaries:
# /{bin,sbin}; /usr/{bin,sbin}

BIN="$MERGED_LISTINGS_DIR/bin"

cat "$SPLIT_LISTINGS_DIR/bin.lst" "$SPLIT_LISTINGS_DIR/sbin.lst" > "$BIN"

cat "$SPLIT_LISTINGS_DIR/usr/bin.lst" "$SPLIT_LISTINGS_DIR/usr/sbin.lst" \
    | sed -e 's/^/usr\//' >> "$BIN"

sort -o "$BIN" "$BIN"

BIN_DEDUPLICATED="$MERGED_LISTINGS_DIR/bin-deduplicated"

cp "$BIN" "$BIN_DEDUPLICATED"

sed -e 's?^bin/??'      \
    -e 's?^sbin/??'     \
    -e 's?^usr/bin/??'  \
    -e 's?^usr/sbin/??' \
    "$BIN_DEDUPLICATED" | sponge "$BIN_DEDUPLICATED"

sort -o "$BIN_DEDUPLICATED" "$BIN_DEDUPLICATED"

uniq "$BIN_DEDUPLICATED" | sponge "$BIN_DEDUPLICATED"

# Libraries
# /lib; /usr/{lib,lib32}

LIB="$MERGED_LISTINGS_DIR/lib"

cat "$SPLIT_LISTINGS_DIR/lib.lst" > "$LIB"

lib="$SPLIT_LISTINGS_DIR/usr/lib"
cat "${lib}.lst" "${lib}32.lst" | sed -e 's/^/usr\//' >> "$LIB"

sort -o "$LIB" "$LIB"

LIB_DEDUPLICATED="$MERGED_LISTINGS_DIR/lib-deduplicated"

cp "$LIB" "$LIB_DEDUPLICATED"

sed -e 's?^lib/??'        \
    -e 's?^usr/lib/??'    \
    -e 's?^usr/lib32/??'  \
    "$LIB_DEDUPLICATED" | sponge "$LIB_DEDUPLICATED"

sort -o "$LIB_DEDUPLICATED" "$LIB_DEDUPLICATED"

uniq "$LIB_DEDUPLICATED" | sponge "$LIB_DEDUPLICATED"

