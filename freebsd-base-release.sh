#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract FreeBSD-14.1-base-release-1 package metadata
# from "packagesite.yaml" file and individual .pkg files

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
%s/"origin".*\("comment"\)/\1/g
%s/"maintainer".*\("licenselogic"\)/\1/g
%s/"pkgsize".*\("desc"\)/\1/g
%s/,"categories".*$//g
%s/^{//g
%s/\n/\r\r/g
%s/,\("comment"\)/\r\1/g
%s/,\("licenselogic"\)/\r\1/g
%s/,\("licenses"\)/\r\1/g
%s/,\("desc"\)/\r\1/g
w
q
EOF

# Split simplified listing file on empty lines with splits named
# after first line of each section of the format "name": "filename"

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

# Generate package name listings

TARGET_DIR="$DEST_DIR/names"
mkdir -pv "$TARGET_DIR"

ls "$DEST_DIR/pkgs" > "$TARGET_DIR/base"

ex -s "$TARGET_DIR/base" << EOF
set verbose=1
%s/\.txt//g
w
q
EOF

cp "$TARGET_DIR/base" "$TARGET_DIR/simplified"

ex -s "$TARGET_DIR/simplified" << EOF
set verbose=1
%s/FreeBSD-//g
w
q
EOF

sort "$TARGET_DIR/simplified" -o "$TARGET_DIR/simplified"

cp -v "$TARGET_DIR/simplified" "$TARGET_DIR/no-man-dbg-dev-lib32"

ex -s "$TARGET_DIR/no-man-dbg-dev-lib32" << EOF
set verbose=1
g/-dbg$/d
g/-dbg-/d
g/-man$/d
g/-dev$/d
g/-dev-/d
g/-lib32$/d
w
q
EOF

# Extract no-man-dbg-dev-lib32 packages into seperate directory

TARGET_DIR="$DEST_DIR/pkgs-no-man-dbg-dev-lib32"
mkdir -pv "$TARGET_DIR"

while IFS= read -r pkgname; do
    ln -vsr "$(find "$DEST_DIR/pkgs" -type f -name "FreeBSD-$pkgname.txt")" "$TARGET_DIR"
done < "$DEST_DIR/names/no-man-dbg-dev-lib32"

# Extract "+MANIFEST" files from all .pkg files

PKG_FILES_DIR=$(realpath "$SRC_DIR/pkgs")
MANIFEST_DIR="$DEST_DIR/pkg-manifests"

mkdir -pv "$MANIFEST_DIR"

cd "$MANIFEST_DIR" || exit

for pkg in "$PKG_FILES_DIR"/*; do
    bsdtar -xvf "$pkg" "+MANIFEST"
    mv "+MANIFEST" "$(basename "$pkg" | sed "s/pkg/manifest/")"
done

cd - || exit

# Extract package file listings

LISTING_DIR="$DEST_DIR/pkg-listings"

mkdir -pv "$LISTING_DIR"

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

# Potential exceptions

# # FreeBSD-certctl-14.1.lst
# # FreeBSD-kernel-generic-mmccam-14.1p2.lst
# # FreeBSD-kernel-minimal-dbg-14.1p2.lst
# # FreeBSD-rescue-14.1p1.lst
# # FreeBSD-runtime-14.1p2.lst
# # FreeBSD-runtime-man-14.1.lst
# # FreeBSD-src-14.1p2.lst
# # FreeBSD-utilities-dbg-14.1p1.lst
# # FreeBSD-src-sys-14.1p2.lst

