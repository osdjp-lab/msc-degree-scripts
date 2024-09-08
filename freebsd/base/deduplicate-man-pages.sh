#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Deduplicate extracted man pages based on md5sum

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    MAN_DIR="$(realpath "$1")"
else
    printf "Usage: %s [MAN_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify MAN_DIR

if ! [ -d "$MAN_DIR" ]; then
    printf "Man directory \"%s\" does not exist or is not a directory\n" "$MAN_DIR"
    exit
fi

########################################

# Temporary files

pkg_tmp="$(mktemp)"
hash_file_tmp="$(mktemp)"
ls_tmp="$(mktemp)"
hash_tmp="$(mktemp)"
save_tmp="$(mktemp)"
delete_tmp="$(mktemp)"

find "$MAN_DIR" -mindepth 1 -type d > "$pkg_tmp"

while IFS= read -r pkg; do

    # Write data to temporary files
    
    find "$pkg" -type f | sort > "$ls_tmp"
    md5sum -- "$pkg"/*  > "$hash_file_tmp"
    md5sum -- "$pkg"/* | awk '{print $1}' | sort -u > "$hash_tmp"

    while IFS= read -r hash; do
        grep -m 1 "$hash" "$hash_file_tmp" | awk '{print $2}' >> "$save_tmp"
    done < "$hash_tmp"

    sort "$save_tmp" -o "$save_tmp"
    comm -23 "$ls_tmp" "$save_tmp" > "$delete_tmp"

    # Delete duplicates
    
    while IFS= read -r file; do
        rm "$file"
    done < "$delete_tmp"

    # Clear temporary files

    true > "$hash_file_tmp"
    true > "$ls_tmp"
    true > "$hash_tmp"
    true > "$save_tmp"
    true > "$delete_tmp"

done < "$pkg_tmp"

# Cleanup

rm "$pkg_tmp" "$hash_file_tmp" "$ls_tmp" "$hash_tmp" "$save_tmp" "$delete_tmp"

