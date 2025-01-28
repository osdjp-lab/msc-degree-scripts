#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Split simplified listing file on empty lines with splits named
# after first line of each section of the format "Package: filename"

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SIMPLIFIED_LISTING="$(realpath "$1")"
    PKG_META_DIR="$(realpath "$2")"
else
    printf "Usage: %s [SIMPLIFIED_LISTING] [PKG_META_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify SIMPLIFIED_LISTING and PKG_META_DIR 

if ! [ -f "$SIMPLIFIED_LISTING" ]; then
    printf "Input file \"%s\" does not exist or is not a regular file\n" "$SIMPLIFIED_LISTING"
    exit
fi

if ! [ -d "$PKG_META_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$PKG_META_DIR"
    exit
fi

########################################

cd "$PKG_META_DIR" || exit

awk -v RS="" '
{
    # Split the record into lines
    split($0, lines, "\n")
    
    # Extract the filename from the first line
    match(lines[1], /^Package: ([^ ]+)/, arr)
    filename = arr[1]

    # Write the record to the file named after the extracted filename
    print $0 > filename ".txt"
    close(filename ".txt")
}' "$SIMPLIFIED_LISTING"

cd - || exit


