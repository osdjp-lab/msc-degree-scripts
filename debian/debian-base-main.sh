#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract debian-bookworm main package metadata
# from "Packages" and "Contents-amd64" files

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
    printf "SRC_DIR does not exist or is not a directory\n"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "DEST_DIR does not exist or is not a directory\n"
    exit
fi

# Check if "Packages" file is present and readable in SRC_DIR

if ! [ -r "$SRC_DIR/Packages" ]; then
    printf "File \"Packages\" in SRC_DIR is not readable or does not exist\n"
    exit
fi

# Check if "Contents-amd64" file is present and readable in SRC_DIR

if ! [ -r "$SRC_DIR/Contents-amd64" ]; then
    printf "File \"Contents-amd64\" in SRC_DIR is not readable or does not exist\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/base"

PKG_LISTING="$SRC_DIR/Packages"
SIMPLIFIED_LISTING="$DEST_DIR/simplified"
"$SCRIPT_DIR/simplify-lst.sh" "$PKG_LISTING" "$SIMPLIFIED_LISTING"

PKG_META_DIR="$DEST_DIR/pkgs"
mkdir -pv "$PKG_META_DIR"
"$SCRIPT_DIR/split-lst.sh" "$SIMPLIFIED_LISTING" "$PKG_META_DIR"

CONTENTS_LISTING="$SRC_DIR/Contents-amd64"
NAME_DIR="$DEST_DIR/names"
mkdir -pv "$NAME_DIR"
"$SCRIPT_DIR/generate-names.sh" "$CONTENTS_LISTING" "$PKG_META_DIR" "$NAME_DIR"
"$SCRIPT_DIR/generate-all-pkg-lst.sh" "$CONTENTS_LISTING" "$NAME_DIR"

LISTING_DIR="$DEST_DIR/pkg-listings"
mkdir -pv "$LISTING_DIR"
PKG_CONTENTS_LISTING="$NAME_DIR/pkg-contents"
cd "$LISTING_DIR" || exit
awk -f "$SCRIPT_DIR/generate-pkg-lst.awk" "$PKG_CONTENTS_LISTING"
cd - || exit

PKGS_BY_PRIORITY_DIR="$DEST_DIR/by-priority"
mkdir -pv "$PKGS_BY_PRIORITY_DIR"
"$SCRIPT_DIR/group-by-priority.sh" "$PKG_META_DIR" "$PKGS_BY_PRIORITY_DIR"

PKGS_BY_SECTION_DIR="$DEST_DIR/by-section"
mkdir -pv "$PKGS_BY_SECTION_DIR"
"$SCRIPT_DIR/group-by-section.sh" "$PKG_META_DIR" "$PKGS_BY_SECTION_DIR"

