#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract debian-bookworm main package metadata
# from "Packages" and "Contents-amd64" files

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

# Check if "Packages" file is present and readable in SRC_DIR

if ! [ -r "$SRC_DIR/Packages" ]; then
    printf "File \"Packages\" in source directory is not readable or does not exist\n"
    exit
fi

# Check if "Contents-amd64" file is present and readable in SRC_DIR

if ! [ -r "$SRC_DIR/Contents-amd64" ]; then
    printf "File \"Contents-amd64\" in source directory is not readable or does not exist\n"
    exit
fi

# Simplify debian package listing

OUTPUT="simplified"

cp -v "$SRC_DIR/Packages" "$DEST_DIR/$OUTPUT"

ex -s "$DEST_DIR/$OUTPUT" << EOF
set verbose=1
%s/^Installed-Size:.*\n//g
%s/^Maintainer:.*\n//g
%s/^Architecture:.*\n//g
%s/^Depends:.*\n//g
%s/^Pre-Depends:.*\n//g
%s/^Homepage:.*\n//g
%s/^Description-md5:.*\n//g
%s/^Filename:.*\n//g
%s/^Size:.*\n//g
%s/^MD5sum:.*\n//g
%s/^SHA256:.*\n//g
%s/^Source:.*\n//g
%s/^Replaces:.*\n//g
%s/^Suggests:.*\n//g
%s/^Breaks:.*\n//g
%s/^Multi-Arch:.*\n//g
%s/^Recommends:.*\n//g
%s/^Enhances:.*\n//g
%s/^Provides:.*\n//g
%s/^Conflicts:.*\n//g
w
q
EOF

# Split simplified listing file on empty lines with splits named after first line of each section of the format "Package: filename"

mkdir -pv "$DEST_DIR/pkgs"
cd "$DEST_DIR/pkgs" || exit

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
}' "../$OUTPUT"

cd - || exit

# Sort split package files by section

SEARCH_DIR="$DEST_DIR/pkgs"
OUTPUT_DIR="$DEST_DIR/pkgs-by-section"
ERROR_DIR="$DEST_DIR/pkg-sort-errors"

mkdir -pv "$OUTPUT_DIR"
mkdir -pv "$ERROR_DIR"

printf "Making section directories\n"

for section in $(grep -rh "Section: " "$SEARCH_DIR" | awk -F': ' '{print $2}' | sort | uniq)
do
    mkdir -pv "$OUTPUT_DIR/$section"
done

printf "Sorting packages\n"

for file in "$SEARCH_DIR"/*
do
    if [ -f "$file" ]; then
        printf "%s\n" "$file"
        section=$(grep -m 1 "Section: " "$file" | awk -F': ' '{print $2}')
        if [ -n "$section" ]; then
            ln -vsr "$file" "$OUTPUT_DIR/$section/"
        else
            ln -vsr "$file" "$ERROR_DIR/"
        fi
    fi
done

rmdir -v "$ERROR_DIR"

# Sort packages by priority

# Priority: required > important > standard > optional > extra

SEARCH_DIR="$DEST_DIR/pkgs"
BASE_DIR="$DEST_DIR/pkgs-by-priority"

TARGET_DIR="$BASE_DIR/required-pkgs"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: required" "$SEARCH_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$BASE_DIR/important-pkgs"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: important" "$SEARCH_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$BASE_DIR/standard-pkgs"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: standard" "$SEARCH_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$BASE_DIR/optional-pkgs"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: optional" "$SEARCH_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$BASE_DIR/extra-pkgs"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: extra" "$SEARCH_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

# Extract package file listings

NAME_FIRST="$DEST_DIR/name-first"

cp -v "$SRC_DIR/Contents-amd64" "$NAME_FIRST"

# Switch columns and replace separator with '\t'

ex -s "$NAME_FIRST" << EOF
%s/^\(.*\)\s\(.*\)$/\2\t\1/g
w
q
EOF

sort "$NAME_FIRST" -o "$NAME_FIRST"

# Make list of output file names in listing

PKG_NAMES="$DEST_DIR/pkg-names"

cut -f1 "$NAME_FIRST" > "$PKG_NAMES"

sort "$PKG_NAMES" -o "$PKG_NAMES"

uniq "$PKG_NAMES" | rev | cut -d'/' -f1 | rev | sponge "$PKG_NAMES"

# Create empty output files

LISTING_DIR="$DEST_DIR/pkg-listings"
mkdir -pv "$LISTING_DIR"

while IFS= read -r line
do
    true > "$LISTING_DIR/$line.lst"
done < "$PKG_NAMES"

# Sort contents of original listing into previously created files

while IFS= read -r line
do
    pkg_name=$(printf "%s" "$line" | cut -f1 | rev | cut -d'/' -f1 | rev)
    printf "%s\n" "$pkg_name"
    printf "%s\n" "$line" | cut -f2 >> "$LISTING_DIR/$pkg_name.lst"
done < "$NAME_FIRST"

