#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Download and extract debian-bookworm main package metadata
# from "Packages" and "Contents-amd64" files

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    DEST_DIR="$(realpath "$1")"
else
    printf "Usage: %s [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify DEST_DIR

if ! [ -d "$DEST_DIR" ]; then
    printf "DEST_DIR does not exist or is not a directory\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/base"

SRC_DIR="$DEST_DIR/src"
if ! [ -d "$SRC_DIR" ]; then
    mkdir -pv "$SRC_DIR"
    "$SCRIPT_DIR/get-data.sh" "$SRC_DIR"
fi

CONTENTS_DIR="$DEST_DIR/Contents-amd64"
mkdir -pv "$CONTENTS_DIR"
"$SCRIPT_DIR/Contents-amd64.sh" "$SRC_DIR/Contents-amd64" "$CONTENTS_DIR"

PACKAGES_DIR="$DEST_DIR/Packages"
mkdir -pv "$PACKAGES_DIR"
"$SCRIPT_DIR/Packages.sh" "$SRC_DIR/Packages" "$PACKAGES_DIR"

