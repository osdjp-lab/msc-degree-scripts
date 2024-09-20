#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group packages by relevance

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKGS_BY_TYPE="$(realpath "$1")"
    PKGS_BY_RELEVANCE="$(realpath "$2")"
else
    printf "Usage: %s [PKGS_BY_TYPE] [PKGS_BY_RELEVANCE]\n" "$(basename "$0")"
    exit
fi

# Verify PKGS_BY_TYPE and PKGS_BY_RELEVANCE

if ! [ -d "$PKGS_BY_TYPE" ]; then
    printf "Pkgs by type directory \"%s\" does not exist or is not a directory\n" "$PKGS_BY_TYPE"
    exit
fi

if ! [ -d "$PKGS_BY_RELEVANCE" ]; then
    printf "Pkgs by relevance directory \"%s\" does not exist or is not a directory\n" "$PKGS_BY_RELEVANCE"
    exit
fi

########################################

# Relevant packages

DEST_DIR="$PKGS_BY_RELEVANCE/relevant"
mkdir -pv "$DEST_DIR"

cp -rv "$PKGS_BY_TYPE/system" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/net" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/util" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/bundle" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/virt" "$DEST_DIR"

# Irrelevant packages

DEST_DIR="$PKGS_BY_RELEVANCE/irrelevant"
mkdir -pv "$DEST_DIR"

cp -rv "$PKGS_BY_TYPE/lib" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/dev-tools" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/data" "$DEST_DIR"
cp -rv "$PKGS_BY_TYPE/other" "$DEST_DIR"

