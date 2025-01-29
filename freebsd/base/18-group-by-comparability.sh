#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group packages by contents comparability

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKGS_BY_TYPE="$(realpath "$1")"
    PKGS_BY_COMPARABILITY="$(realpath "$2")"
else
    printf "Usage: %s [PKGS_BY_TYPE] [PKGS_BY_COMPARABILITY]\n" "$(basename "$0")"
    exit
fi

# Verify PKGS_BY_TYPE and PKGS_BY_COMPARABILITY

if ! [ -d "$PKGS_BY_TYPE" ]; then
    printf "Pkgs by type directory \"%s\" does not exist or is not a directory\n" "$PKGS_BY_TYPE"
    exit
fi

if ! [ -d "$PKGS_BY_COMPARABILITY" ]; then
    printf "Pkgs by comparability directory \"%s\" does not exist or is not a directory\n" "$PKGS_BY_COMPARABILITY"
    exit
fi

########################################

# Comparable packages

DEST_DIR="$PKGS_BY_COMPARABILITY/comparable"
mkdir -pv "$DEST_DIR"

cp -rv "$PKGS_BY_TYPE/bundle" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/data" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/dev-tools" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/lib" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/util" "$DEST_DIR"

# Non-comparable packages

DEST_DIR="$PKGS_BY_COMPARABILITY/non-comparable"
mkdir -pv "$DEST_DIR"

cp -rv "$PKGS_BY_TYPE/other" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/system" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/virt" "$DEST_DIR"

# Unknown

DEST_DIR="$PKGS_BY_COMPARABILITY/unknown"
mkdir -pv "$DEST_DIR"

cp -rv "$PKGS_BY_TYPE/net" "$DEST_DIR"

