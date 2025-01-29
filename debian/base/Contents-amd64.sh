#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract debian-bookworm main package metadata
# from "Contents-amd64" file

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    CONTENTS_LISTING="$(realpath "$1")"
    DEST_DIR="$(realpath "$2")"
else
    printf "Usage: %s [CONTENTS_LISTING] [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify CONTENTS_LISTING and DEST_DIR

if ! [ -r "$CONTENTS_LISTING" ]; then
    printf "File CONTENTS_LISTING is not readable or does not exist\n"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "DEST_DIR does not exist or is not a directory\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/Contents-amd64"

NAME_DIR="$DEST_DIR/names"
mkdir -pv "$NAME_DIR"

PKG_CONTENTS_LISTING="$NAME_DIR/pkg-contents"
"$SCRIPT_DIR/1-generate-contents-lst.sh" "$CONTENTS_LISTING" "$PKG_CONTENTS_LISTING"

PKG_LISTING="$NAME_DIR/pkgs"
"$SCRIPT_DIR/2-generate-pkg-lst.sh" "$PKG_CONTENTS_LISTING" "$PKG_LISTING"

ALL_PKG_LISTING="$NAME_DIR/all-pkg.lst"
"$SCRIPT_DIR/3-generate-all-pkg-lst.sh" "$CONTENTS_LISTING" "$ALL_PKG_LISTING"

SPLIT_LISTINGS_DIR="$DEST_DIR/split-lst"
mkdir -pv "$SPLIT_LISTINGS_DIR"
"$SCRIPT_DIR/4-split-lst-by-dir.sh" "$ALL_PKG_LISTING" "$SPLIT_LISTINGS_DIR"

MERGED_LISTINGS_DIR="$DEST_DIR/merged-lst"
mkdir -pv "$MERGED_LISTINGS_DIR"
"$SCRIPT_DIR/5-generate-merged-dir-lst.sh" "$SPLIT_LISTINGS_DIR" "$MERGED_LISTINGS_DIR"

LISTING_DIR="$DEST_DIR/pkg-listings"
mkdir -pv "$LISTING_DIR"
cd "$LISTING_DIR" || exit
awk -f "$SCRIPT_DIR/6-generate-pkg-lst.awk" "$PKG_CONTENTS_LISTING"
cd - || exit

