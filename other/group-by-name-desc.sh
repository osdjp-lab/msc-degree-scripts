#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group packages from FreeBSD base and Debian bookworm main
# on the basis of package names and descriptions

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

# Check if required files are present and readable in SRC_DIR subdirectories

FREEBSD_BASE_LISTING="$SRC_DIR/freebsd/base/names/no-man-dbg-dev-lib32"

if ! [ -r "$FREEBSD_BASE_LISTING" ]; then
    printf "File \"freebsd/base/names/no-man-dbg-dev-lib32\" in source directory is not readable or does not exist\n"
    exit
fi

# Sort debian package metadata files based on freebsd base package files

FREEBSD_BASE_PKGS="$SRC_DIR/freebsd/base/pkgs"
DEBIAN_PKGS="$SRC_DIR/debian/pkgs"

# Make directories for all packages

MATCH_FAIL="$DEST_DIR/no-match-pkgs"

mkdir -pv "$MATCH_FAIL"

while IFS= read -r packagename; do
    
    # Search package names
    
    NAME_MATCHES=$(find "$DEBIAN_PKGS" -type f -name "*$packagename*")
    
    if [ -n "$NAME_MATCHES" ]
    then
        echo "Path 1"
        mkdir -pv "$packagename"
        
        # Copy freebsd package metadata file
        
        cp -vt "$packagename/" "$FREEBSD_BASE_PKGS/FreeBSD-$packagename.txt"
        
        # Copy matching debian package metadata files
        
        mkdir -pv "$packagename/debian-name-match"
        printf "%s" "$NAME_MATCHES" | xargs cp -vt "$packagename/debian-name-match"
    fi
    
    # Search package descriptions
    
    DESC_MATCHES=$(grep -rli "Description:.*$packagename.*" "$DEBIAN_PKGS")
    
    if [ -n "$DESC_MATCHES" ]
    then
        echo "Path 2"
        mkdir -pv "$packagename"
        
        # Copy freebsd package meta file to directory
        
        cp -vt "$packagename/" "$FREEBSD_BASE_PKGS/FreeBSD-$packagename.txt"
        
        # Grep for packages with same word in debian packages
        
        mkdir -pv "$packagename/debian-desc-match"
        printf "%s" "$DESC_MATCHES" | xargs cp -vt "$packagename/debian-desc-match"
    else
        echo "Path 3"
        cp -vt "$MATCH_FAIL" "$FREEBSD_BASE_PKGS/FreeBSD-$packagename.txt"
    fi
done < "$FREEBSD_BASE_LISTING"

rmdir -v "$MATCH_FAIL"

