#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract FreeBSD-14.1-release-1 package metadata
# from "packagesite.yaml" file

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SRC_DIR="$(realpath "$1")"
    DEST_DIR="$(realpath "$2")"
else
    printf "Usage: %s [SRC_DIR] [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify values of SRC_DIR and DEST_DIR

if ! [ -d "$SRC_DIR" ]; then
    printf "Source \"%s\" does not exist or is not a directory\n" "$SRC_DIR"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$DEST_DIR"
    exit
fi

# Check if "packagesite.yaml" file is present and readable in SRC_DIR

if ! [ -r "$SRC_DIR/packagesite.yaml" ]; then
    printf "File \"packagesite.yaml\" in SRC_DIR is not readable or does not exist\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/release"

BASE_LISTING="$SRC_DIR/packagesite.yaml" 
SIMPLIFIED_LISTING="$DEST_DIR/simplified"
"$SCRIPT_DIR/simplify-lst.sh" "$BASE_LISTING" "$SIMPLIFIED_LISTING"

PKG_META_DIR="$DEST_DIR/pkgs"
mkdir -pv "$PKG_META_DIR"
"$SCRIPT_DIR/split-lst.sh" "$SIMPLIFIED_LISTING" "$PKG_META_DIR"

NAME_DIR="$DEST_DIR/names"
mkdir -pv "$NAME_DIR"
"$SCRIPT_DIR/generate-names.sh" "$NAME_DIR" "$PKG_META_DIR"

