#!/usr/bin/env sh

set -x

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract debian-bookworm main package metadata
# from "Packages" file

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_LISTING="$(realpath "$1")"
    DEST_DIR="$(realpath "$2")"
else
    printf "Usage: %s [PKG_LISTING] [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_LISTING and DEST_DIR

if ! [ -f "$PKG_LISTING" ]; then
    printf "Input file \"%s\" does not exist or is not a regular file\n" "$PKG_LISTING"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "DEST_DIR does not exist or is not a directory\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/Packages"

SIMPLIFIED_LISTING="$DEST_DIR/simplified"
"$SCRIPT_DIR/1-simplify-lst.sh" "$PKG_LISTING" "$SIMPLIFIED_LISTING"

PKG_META_DIR="$DEST_DIR/pkgs"
mkdir -pv "$PKG_META_DIR"
"$SCRIPT_DIR/2-split-lst.sh" "$SIMPLIFIED_LISTING" "$PKG_META_DIR"

NAME_DIR="$DEST_DIR/names"
mkdir -pv "$NAME_DIR"
"$SCRIPT_DIR/3-generate-names.sh" "$PKG_META_DIR" "$NAME_DIR"

PKGS_BY_PRIORITY_DIR="$DEST_DIR/by-priority"
mkdir -pv "$PKGS_BY_PRIORITY_DIR"
"$SCRIPT_DIR/4-group-by-priority.sh" "$PKG_META_DIR" "$PKGS_BY_PRIORITY_DIR"

PKGS_BY_SECTION_DIR="$DEST_DIR/by-section"
mkdir -pv "$PKGS_BY_SECTION_DIR"
"$SCRIPT_DIR/5-group-by-section.sh" "$PKG_META_DIR" "$PKGS_BY_SECTION_DIR"

