#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group packages by section

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_META_DIR="$(realpath "$1")"
    PKGS_BY_SECTION_DIR="$(realpath "$2")"
else
    printf "Usage: %s [PKG_META_DIR] [PKGS_BY_SECTION_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_META_DIR and PKGS_BY_PRIORITY

if ! [ -d "$PKG_META_DIR" ]; then
    printf "PKG_META_DIR does not exist or is not a directory\n"
    exit
fi

if ! [ -d "$PKGS_BY_SECTION_DIR" ]; then
    printf "PKGS_BY_SECTION_DIR does not exist or is not a directory\n"
    exit
fi

########################################

# Sort split package files by section

ERROR_DIR="$PKGS_BY_SECTION_DIR/pkg-sort-errors"
mkdir -pv "$ERROR_DIR"

printf "Making section directories\n"

for section in $(grep -rh "Section: " "$PKG_META_DIR" | awk -F': ' '{print $2}' | sort | uniq)
do
    mkdir -pv "$PKGS_BY_SECTION_DIR/$section"
done

printf "Sorting packages\n"

for file in "$PKG_META_DIR"/*
do
    if [ -f "$file" ]; then
        printf "%s\n" "$file"
        section=$(grep -m 1 "Section: " "$file" | awk -F': ' '{print $2}')
        if [ -n "$section" ]; then
            ln -vsr "$file" "$PKGS_BY_SECTION_DIR/$section/"
        else
            ln -vsr "$file" "$ERROR_DIR/"
        fi
    fi
done

rmdir -v "$ERROR_DIR"

