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

# Extract "+MANIFEST" files from all .pkg files

PKG_FILES_DIR=$(realpath "$SRC_DIR/pkgs")
MANIFEST_DIR="$DEST_DIR/pkg-manifests"

mkdir -pv "$MANIFEST_DIR"

cd "$MANIFEST_DIR" || exit

for pkg in "$PKG_FILES_DIR"/*; do
    bsdtar -xvf "$pkg" "+MANIFEST"
    mv "+MANIFEST" "$(basename "$pkg" | sed "s/\.pkg/\.manifest/")"
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

find "$MANIFEST_DIR" -type f -name "*.manifest" -delete
rmdir "$MANIFEST_DIR"

# Create deduplicated single file all base system package file listing

ALL_PKG_LST="$DEST_DIR/all-pkg.lst"

cat "$LISTING_DIR"/* > "$ALL_PKG_LST"

ex -s "$ALL_PKG_LST" << EOF
g/"files":/d
g/"config":/d
g/"directories":/d
g/^$/d
%s/:"y"//g
%s/"//g
%s/^\///g
w
q
EOF

sort "$ALL_PKG_LST" -o "$ALL_PKG_LST"

uniq "$ALL_PKG_LST" | sponge "$ALL_PKG_LST"

# Extract package man pages

MAN_DIR="$DEST_DIR/pkg-man-pages"
mkdir -pv "$MAN_DIR"

TMP_DIR="$DEST_DIR/tmp"
mkdir "$TMP_DIR"

find "$PKG_FILES_DIR" -name "*man*" > "$DEST_DIR/man-pkgs"

cd "$TMP_DIR" || exit

while IFS= read -r pkg; do
    bsdtar -xvf "$pkg"
    rm "+MANIFEST" "+COMPACT_MANIFEST"
    pkg_name="$(basename "$pkg" | sed "s/FreeBSD-\(.*\)-man.*/\1/")"
    mkdir -pv "$MAN_DIR/$pkg_name"
    find . -type f -exec mv -vt "$MAN_DIR/$pkg_name" {} +
done < "$DEST_DIR/man-pkgs"

cd - || exit

rm "$DEST_DIR/man-pkgs"

find "$TMP_DIR" -type d -empty -delete

# Deduplicate extracted man pages based on md5sum

# Temporary files

pkg_tmp="$(mktemp)"
hash_file_tmp="$(mktemp)"
ls_tmp="$(mktemp)"
hash_tmp="$(mktemp)"
save_tmp="$(mktemp)"
delete_tmp="$(mktemp)"

find "$MAN_DIR" -mindepth 1 -type d > "$pkg_tmp"

while IFS= read -r pkg; do

    # Writing data to temporary files
    find "$pkg" -type f | sort > "$ls_tmp"
    md5sum -- "$pkg"/*  > "$hash_file_tmp"
    md5sum -- "$pkg"/* | awk '{print $1}' | sort -u > "$hash_tmp"

    while IFS= read -r hash; do
        grep -m 1 "$hash" "$hash_file_tmp" | awk '{print $2}' >> "$save_tmp"
    done < "$hash_tmp"

    sort "$save_tmp" -o "$save_tmp"
    comm -23 "$ls_tmp" "$save_tmp" > "$delete_tmp"

    # Deleting duplicates

    while IFS= read -r file; do
        rm "$file"
    done < "$delete_tmp"

    # Clearing temporary files

    true > "$hash_file_tmp"
    true > "$ls_tmp"
    true > "$hash_tmp"
    true > "$save_tmp"
    true > "$delete_tmp"

done < "$pkg_tmp"

# Cleanup

rm "$pkg_tmp" "$hash_file_tmp" "$ls_tmp" "$hash_tmp" "$save_tmp" "$delete_tmp"

# Extract plain text man pages

PLAIN_MAN_DIR="$DEST_DIR/pkg-man-txt"
cp -rv "$MAN_DIR" "$PLAIN_MAN_DIR"

pkg_man_dir_tmp="$(mktemp)"
pkg_man_tmp="$(mktemp)"

find "$PLAIN_MAN_DIR" -mindepth 1 -type d > "$pkg_man_dir_tmp"

export MANWIDTH=10000

while IFS= read -r pkg; do
    find "$pkg" -type f > "$pkg_man_tmp"
    while IFS= read -r man_page; do
        printf "%s\n" "$man_page"
        pkg_basename="$(man -l "$man_page" | head -n 1 | cut -d' ' -f 1 | sed -e 's/^\(.*\)(\(.*\))$/\1-\2/' | tr '[:upper:]' '[:lower:]')"
        if [ -z "$pkg_basename" ]; then
            pkg_basename="$(man -l "$man_page" | head -n 2 | tail -n 1 | cut -d' ' -f 1 | sed -e 's/^\(.*\)(\(.*\))$/\1-\2/' | tr '[:upper:]' '[:lower:]')"
        fi
        dest_file_name="$PLAIN_MAN_DIR/$(basename "$pkg")/$pkg_basename"
        dest_file="${dest_file_name}-$(cksum "$man_page" | cut -d' ' -f1).txt"
        man -l "$man_page" 2>/dev/null | awk '/NAME/,/^[[:blank:]]*$/; /DESCRIPTION/,/^[[:blank:]]*$/' > "$dest_file"
        rm "$man_page"
    done < "$pkg_man_tmp"
done < "$pkg_man_dir_tmp"

unset MANWIDTH

rm "$pkg_man_dir_tmp" "$pkg_man_tmp"

