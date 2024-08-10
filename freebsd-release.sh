#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract FreeBSD-14.1-release-1 package metadata
# from "packagesite.yaml" file

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    SRC_DIR="$(realpath "$1")"
    DEST_DIR="$(realpath "$2")"
else
    printf "Usage: %s [SRC_DIR] [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify values of SRC_DIR and DEST_DIR

if ! [ -d "$SRC_DIR" ]; then
    printf "Source \"%s\" does not exist or is not a directory\n" "$SRC_DIR"
    exit
fi

if ! [ -d "$DEST_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$DEST_DIR"
    exit
fi

# Check if "packagesite.yaml" file is present and readable in SRC_DIR

if ! [ -r "$SRC_DIR/packagesite.yaml" ]; then
    printf "File \"packagesite.yaml\" in source directory is not readable or does not exist\n"
    exit
fi

# Simplify freebsd package listing

OUTPUT="simplified"

cp -v "$SRC_DIR/packagesite.yaml" "$DEST_DIR/$OUTPUT"

ex -s "$DEST_DIR/$OUTPUT" << EOF
set verbose=1
%s/"origin".*\("version".*"comment"\)/\1/g
%s/"maintainer".*\("licenselogic"\)/\1/g
%s/"pkgsize".*\("desc"\)/\1/g
%s/,"options".*$//g
%s/,"shlibs.*$//g
%s/"deps".*\("categories"\)/\1/g
%s/,"annotations".*$//g
%s/,"users".*$//g
%s/,"groups".*$//g
%s/^{//g
%s/\n/\r\r/g
%s/,\("version"\)/\r\1/g
%s/,\("comment"\)/\r\1/g
%s/,\("licenselogic"\)/\r\1/g
%s/,\("licenses"\)/\r\1/g
%s/,\("desc"\)/\r\1/g
%s/,\("categories"\)/\r\1/g
w
q
EOF

# Split simplified listing file on empty lines with splits named after first line of each section of the format "name": "filename"

mkdir -pv "$DEST_DIR/pkgs"
cd "$DEST_DIR/pkgs" || exit

awk -v RS="" '
{
    # Split the record into lines
    split($0, lines, "\n")
    
    # Extract the filename from the first line
    match(lines[1], /^"name":"([^ ]+)"/, arr)
    filename = arr[1]

    # Write the record to the file named after the extracted filename
    print $0 > filename ".txt"
    close(filename ".txt")
}' "../$OUTPUT"

cd - || exit

# Generate package name listing

ls "$DEST_DIR/pkgs" > "$DEST_DIR/names"

ex -s "$DEST_DIR/names" << EOF
set verbose=1
%s/\.txt//g
w
q
EOF

# Generate category listing

grep -ri '"categories"' "$DEST_DIR/pkgs" > "$DEST_DIR/categories"

ex -s "$DEST_DIR/categories" << EOF
set verbose=1
%s/^.*categories":\[//g
%s/\]//g
%s/,/\r/g
%s/"//g
w
q
EOF

sort -u "$DEST_DIR/categories" -o "$DEST_DIR/categories"

