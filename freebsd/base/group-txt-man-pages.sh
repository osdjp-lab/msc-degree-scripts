#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group plain text man pages by category

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    PLAIN_MAN_DIR="$(realpath "$1")"
else
    printf "Usage: %s [PLAIN_MAN_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PLAIN_MAN_DIR

if ! [ -d "$PLAIN_MAN_DIR" ]; then
    printf "Plain man files directory \"%s\" does not exist or is not a directory\n" "$PLAIN_MAN_DIR"
    exit
fi

########################################

for pkg_dir in "$PLAIN_MAN_DIR"/*; do
    printf "%s\n" "$(basename "$pkg_dir")"
    for category in $(seq 1 9); do
        mkdir -p "$pkg_dir/$category"
        find "$pkg_dir" -maxdepth 1 -type f -name "*${category}.txt" -exec mv '{}' "$pkg_dir/$category/" \;
    done
    mkdir -p "$pkg_dir/other"
    find "$pkg_dir" -maxdepth 1 -type f -exec mv '{}' "$pkg_dir/other/" \;
    find "$pkg_dir" -type d -empty -delete
done

