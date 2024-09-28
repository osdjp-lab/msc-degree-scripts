#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Compare package file contents listings from FreeBSD base
# and Debian bookworm main 

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

# Check if "all-pkg.lst" files are present and readable in SRC_DIR subdirectories

DEBIAN_LST="$SRC_DIR/debian/names/all-pkg.lst"
FREEBSD_LST="$SRC_DIR/freebsd/base/names/all-pkg.lst"

if ! [ -r "$DEBIAN_LST" ]; then
    printf "File \"debian/all-pkg.lst\" in source directory is not readable or does not exist\n"
    exit
fi

if ! [ -r "$FREEBSD_LST" ]; then
    printf "File \"freebsd/base/all-pkg.lst\" in source directory is not readable or does not exist\n"
    exit
fi

# Generate initial comparison results

TARGET_DIR="$DEST_DIR/by-contents"

INITIAL_DIR="$TARGET_DIR/initial"
mkdir -pv "$INITIAL_DIR"

comm -12 "$FREEBSD_LST" "$DEBIAN_LST" > "$INITIAL_DIR/common"
comm -23 "$FREEBSD_LST" "$DEBIAN_LST" > "$INITIAL_DIR/freebsd-only"
comm -13 "$FREEBSD_LST" "$DEBIAN_LST" > "$INITIAL_DIR/debian-only"

# Recover debian package data for common

DEBIAN_PKG_CONTENTS="$SRC_DIR/debian/names/pkg-contents"
MATCHED_DIR="$TARGET_DIR/matched"
mkdir -pv "$MATCHED_DIR"

MATCHED_DEBIAN_PKGS="$MATCHED_DIR/debian-pkgs"

true > "$MATCHED_DEBIAN_PKGS"

while IFS= read -r file; do
    grep -F "$file" "$DEBIAN_PKG_CONTENTS" | cut -f1 >> "$MATCHED_DEBIAN_PKGS"
done < "$INITIAL_DIR/common"

sort "$MATCHED_DEBIAN_PKGS" -o "$MATCHED_DEBIAN_PKGS"

uniq "$MATCHED_DEBIAN_PKGS" | sponge "$MATCHED_DEBIAN_PKGS"

# Filter unmatched files for matched debian packages

MATCHED_DEBIAN_PKGS_ALL_FILES="$MATCHED_DIR/all-files"
MATCHED_DEBIAN_PKGS_UNMATCHED_FILES="$MATCHED_DIR/unmatched-files"
MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS="$MATCHED_DIR/unmatched-pkg-file-sets"

true > "$MATCHED_DEBIAN_PKGS_ALL_FILES"
true > "$MATCHED_DEBIAN_PKGS_UNMATCHED_FILES"
true > "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS"

while IFS= read -r pkg; do
    grep -F "$pkg" "$DEBIAN_PKG_CONTENTS" | cut -f2 >> "$MATCHED_DEBIAN_PKGS_ALL_FILES"
done < "$MATCHED_DEBIAN_PKGS"

sort "$MATCHED_DEBIAN_PKGS_ALL_FILES" -o "$MATCHED_DEBIAN_PKGS_ALL_FILES"

comm -12 "$MATCHED_DEBIAN_PKGS_ALL_FILES" "$INITIAL_DIR/debian-only" > "$MATCHED_DEBIAN_PKGS_UNMATCHED_FILES"

while IFS= read -r file; do
    grep -F "$file" "$DEBIAN_PKG_CONTENTS" >> "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS"
done < "$MATCHED_DEBIAN_PKGS"

sort "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS" -o "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS"

