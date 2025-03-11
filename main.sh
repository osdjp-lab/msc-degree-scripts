#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Run all extraction and comparison scripts

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    DEST_DIR="$(realpath "$1")"
else
    printf "Usage: %s [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify value of DEST_DIR

if ! [ -d "$DEST_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$DEST_DIR"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"

EXTRACT="$DEST_DIR/extract"
mkdir -pv "$EXTRACT"

mkdir -pv "$EXTRACT/debian"
"$WORK_DIR/debian/debian-base-main.sh" "$EXTRACT/debian"

mkdir -pv "$EXTRACT/freebsd/base"
"$WORK_DIR/freebsd/freebsd-base-release.sh" "$EXTRACT/freebsd/base"

mkdir -pv "$EXTRACT/freebsd/release"
"$WORK_DIR/freebsd/freebsd-release.sh" "$EXTRACT/freebsd/release"

COMPARE="$DEST_DIR/compare"
mkdir -pv "$COMPARE"

mkdir -pv "$COMPARE/by-name"
"$WORK_DIR/compare/by-name.sh" "$EXTRACT/debian/Contents-amd64/names/pkgs" \
    "$EXTRACT/freebsd/base/names/no-man-dbg-dev-lib32" \
    "$COMPARE/by-name"

mkdir -pv "$COMPARE/by-contents"
"$WORK_DIR/compare/by-contents.sh" "$EXTRACT/freebsd/base/merged-lst/" \
    "$EXTRACT/debian/Contents-amd64/merged-lst/" \
    "$EXTRACT/freebsd/base/names/pkg-contents" \
    "$EXTRACT/debian/Contents-amd64/names/pkg-contents" \
    "$COMPARE/by-contents"

