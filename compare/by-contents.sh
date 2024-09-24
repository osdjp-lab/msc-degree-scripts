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

# Check if "all-pkg.lst" file is present and readable in SRC_DIR subdirectories

DEBIAN_LST="$SRC_DIR/debian/all-pkg.lst"
FREEBSD_LST="$SRC_DIR/freebsd/base/all-pkg.lst"

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
mkdir -pv "$TARGET_DIR"

comm -12 "$FREEBSD_LST" "$DEBIAN_LST" > "$TARGET_DIR/common"
comm -23 "$FREEBSD_LST" "$DEBIAN_LST" > "$TARGET_DIR/freebsd-only"
comm -13 "$FREEBSD_LST" "$DEBIAN_LST" > "$TARGET_DIR/debian-only"

# Recover debian package data for common

MATCHED_DEBIAN_PKGS="$TARGET_DIR/matched-pkgs"

true > "$MATCHED_DEBIAN_PKGS"

while IFS= read -r file; do
    grep "$file" "$SRC_DIR/debian/name-first" | cut -f1 >> "$MATCHED_DEBIAN_PKGS"
done < "$TARGET_DIR/common"

sort "$MATCHED_DEBIAN_PKGS" -o "$MATCHED_DEBIAN_PKGS"

uniq "$MATCHED_DEBIAN_PKGS" | sponge "$MATCHED_DEBIAN_PKGS"

# Filter unmatched files for matched debian packages

MATCHED_DEBIAN_PKGS_ALL_FILES="$TARGET_DIR/matched-pkgs-all-files"
MATCHED_DEBIAN_PKGS_UNMATCHED_FILES="$TARGET_DIR/matched-pkgs-unmatched-files"
MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS="$TARGET_DIR/matched-pkg-unmatched-file-sets"

true > "$MATCHED_DEBIAN_PKGS_ALL_FILES"
true > "$MATCHED_DEBIAN_PKGS_UNMATCHED_FILES"
true > "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS"

while IFS= read -r pkg; do
    grep "$pkg" "$SRC_DIR/debian/name-first" | cut -f2 >> "$MATCHED_DEBIAN_PKGS_ALL_FILES"
done < "$MATCHED_DEBIAN_PKGS"

sort "$MATCHED_DEBIAN_PKGS_ALL_FILES" -o "$MATCHED_DEBIAN_PKGS_ALL_FILES"

comm -12 "$MATCHED_DEBIAN_PKGS_ALL_FILES" "$TARGET_DIR/debian-only" > "$MATCHED_DEBIAN_PKGS_UNMATCHED_FILES"

while IFS= read -r file; do
    grep "$file" "$SRC_DIR/debian/name-first" >> "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS"
done < "$MATCHED_DEBIAN_PKGS"

sort "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS" -o "$MATCHED_DEBIAN_PKG_UNMATCHED_FILE_SETS"

