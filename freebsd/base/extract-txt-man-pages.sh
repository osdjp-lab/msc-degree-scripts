#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract plain text man pages

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    MAN_DIR="$(realpath "$1")"
    PLAIN_MAN_DIR="$(realpath "$2")"
else
    printf "Usage: %s [MAN_DIR] [PLAIN_MAN_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify MAN_DIR and PLAIN_MAN_DIR

if ! [ -d "$MAN_DIR" ]; then
    printf "Man directory \"%s\" does not exist or is not a directory\n" "$MAN_DIR"
    exit
fi

if ! [ -d "$PLAIN_MAN_DIR" ]; then
    printf "Plain man files directory \"%s\" does not exist or is not a directory\n" "$PLAIN_MAN_DIR"
    exit
fi

########################################

cp -Rv "$MAN_DIR"/. "$PLAIN_MAN_DIR"

pkg_man_dir_tmp="$(mktemp)"
pkg_man_tmp="$(mktemp)"

find "$PLAIN_MAN_DIR" -mindepth 1 -type d > "$pkg_man_dir_tmp"

while IFS= read -r pkg; do
    find "$pkg" -type f > "$pkg_man_tmp"
    while IFS= read -r man_page; do
        printf "%s\n" "$man_page"
        man_page_basename="$(basename "$man_page" | sed 's/gz/txt/')"
        dest_file="$PLAIN_MAN_DIR/$(basename "$pkg")/$man_page_basename"
        zcat "$man_page" | groff -t -e -mandoc -Tascii 2>/dev/null | col -bx > "$dest_file"
        rm "$man_page"
    done < "$pkg_man_tmp"
done < "$pkg_man_dir_tmp"

rm "$pkg_man_dir_tmp" "$pkg_man_tmp"

