#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract package file listings

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_FILES_DIR="$(realpath "$1")"
    LISTING_DIR="$(realpath "$2")"
else
    printf "Usage: %s [PKG_FILES_DIR] [LISTING_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_FILES_DIR and LISTING_DIR

if ! [ -d "$PKG_FILES_DIR" ]; then
    printf "Input package files directory \"%s\" does not exist or is not a directory\n" "$PKG_FILES_DIR"
    exit
fi

if ! [ -d "$LISTING_DIR" ]; then
    printf "Output listing files directory \"%s\" does not exist or is not a directory\n" "$LISTING_DIR"
    exit
fi

########################################

# Extract "+MANIFEST" files from all .pkg files

MANIFEST_DIR="$(mktemp -d)"

mkdir -pv "$MANIFEST_DIR"

cd "$MANIFEST_DIR" || exit

for pkg in "$PKG_FILES_DIR"/*; do
    bsdtar -xvf "$pkg" "+MANIFEST"
    mv "+MANIFEST" "$(basename "$pkg" | sed "s/\.pkg/\.manifest/")"
done

cd - || exit

# Process manifest files into pkg file lists

for manifest in "$MANIFEST_DIR"/*; do
    cp "$manifest" "$LISTING_DIR/$(basename "$manifest" | sed "s/manifest/lst/")"
done

for file in "$LISTING_DIR"/*; do
    ex -s "$file" << EOF
set verbose=1
s/^{.*\("files"\)/\1/
s/:"[0-9a-zA-Z$]\{66\}"//g
s/\(["}\]]\),"/\1\r"/g
%s/{/\r/
%s/:\[/:\r/
%s/}//g
%s/\]//g
%s/\("files":\)/\1\r/g
%s/\("config":\)/\r\1\r/g
%s/\("directories":\)/\r\1\r/g
%s/"scripts":.*\n//g
%s/"post-install":.*\n//g
w
q
EOF
done

find "$MANIFEST_DIR" -type f -name "*.manifest" -delete
rmdir "$MANIFEST_DIR"

