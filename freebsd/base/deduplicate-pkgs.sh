#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract deduplicated packages into separate directory

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    NAME_DIR="$(realpath "$1")"
    DEDUPLICATED_PKGS="$(realpath "$2")"
else
    printf "Usage: %s [NAME_DIR] [DEDUPLICATED_PKGS]\n" "$(basename "$0")"
    exit
fi

# Verify NAME_DIR and DEDUPLICATED_PKGS

if ! [ -d "$NAME_DIR" ]; then
    printf "Name directory \"%s\" does not exist or is not a directory\n" "$NAME_DIR"
    exit
fi

if ! [ -d "$DEDUPLICATED_PKGS" ]; then
    printf "Deduplicated pkgs directory \"%s\" does not exist or is not a directory\n" "$DEDUPLICATED_PKGS"
    exit
fi

########################################

DEDUPLICATED_LISTING="$NAME_DIR/no-man-dbg-dev-lib32"

while IFS= read -r pkgname; do
    ln -vsr "$(find "$ALL_PKGS" -type f -name "FreeBSD-$pkgname.txt")" "$DEDUPLICATED_PKGS"
done < "$DEDUPLICATED_LISTING"

