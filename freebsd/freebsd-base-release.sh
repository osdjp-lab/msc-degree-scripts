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
    printf "File \"packagesite.yaml\" in SRC_DIR is not readable or does not exist\n"
    exit
fi

# Check if "pkgs" directory exists in SRC_DIR

if ! [ -d "$SRC_DIR/pkgs" ]; then
    printf "Package subdirectory \"pkgs\" in SRC_DIR does not exist or is not a directory\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/base"

BASE_LISTING="$SRC_DIR/packagesite.yaml" 
SIMPLIFIED_LISTING="$DEST_DIR/simplified"
"$SCRIPT_DIR/simplify-lst.sh" "$BASE_LISTING" "$SIMPLIFIED_LISTING"

PKG_META_DIR="$DEST_DIR/pkgs"
mkdir -pv "$PKG_META_DIR"
"$SCRIPT_DIR/split-lst.sh" "$SIMPLIFIED_LISTING" "$PKG_META_DIR"

PKG_FILES_DIR="$SRC_DIR/pkgs"
LISTING_DIR="$DEST_DIR/pkg-listings"
mkdir -pv "$LISTING_DIR"
"$SCRIPT_DIR/generate-pkg-lst.sh" "$PKG_FILES_DIR" "$LISTING_DIR"

SIMPLIFIED_LISTING_DIR="$DEST_DIR/deduplicated-simplified-pkg-listings"
mkdir -pv "$SIMPLIFIED_LISTING_DIR"
"$SCRIPT_DIR/generate-simplified-pkg-lst.sh" "$LISTING_DIR" "$SIMPLIFIED_LISTING_DIR"

ALL_PKG_LISTING="$NAME_DIR/all-pkg.lst"
"$SCRIPT_DIR/generate-all-pkg-lst.sh" "$LISTING_DIR" "$ALL_PKG_LISTING"

SPLIT_LISTING_DIR="$SRC_DIR/split-lst"
mkdir -pv "$SPLIT_LISTING_DIR"
cd "$SPLIT_LISTING_DIR" || exit
awk -f "$SCRIPT_DIR/generate-split-lst.awk" "$ALL_PKG_LISTING"
cd - || exit

usr_tmp="$(mktemp)"
cp "$SPLIT_LISTING_DIR/usr.lst" "$usr_tmp"

ex -s "$usr_tmp" << EOF
%s/^usr\///g
w
q
EOF

USR_SPLIT_LISTING_DIR="$SPLIT_LISTING_DIR/usr"
mkdir -pv "$USR_SPLIT_LISTING_DIR"
cd "$USR_SPLIT_LISTING_DIR" || exit
awk -f "$SCRIPT_DIR/generate-split-lst.awk" "$usr_tmp"
cd - || exit

rm "$usr_tmp"

MAN_DIR="$DEST_DIR/pkg-man-pages"
mkdir -pv "$MAN_DIR"
"$SCRIPT_DIR/extract-man-pages.sh" "$PKG_FILES_DIR" "$MAN_DIR"
"$SCRIPT_DIR/deduplicate-man-pages.sh" "$MAN_DIR"

PLAIN_MAN_DIR="$DEST_DIR/pkg-man-txt"
mkdir -pv "$PLAIN_MAN_DIR"
"$SCRIPT_DIR/extract-txt-man-pages.sh" "$MAN_DIR" "$PLAIN_MAN_DIR"
"$SCRIPT_DIR/group-txt-man-pages.sh" "$PLAIN_MAN_DIR"

NAME_DIR="$DEST_DIR/names"
mkdir -pv "$NAME_DIR"
"$SCRIPT_DIR/generate-names.sh" "$NAME_DIR" "$PKG_META_DIR" "$MAN_DIR"

DEDUPLICATED_PKGS="$DEST_DIR/pkgs-no-man-dbg-dev-lib32"
mkdir -pv "$DEDUPLICATED_PKGS"
"$SCRIPT_DIR/deduplicate-pkgs.sh" "$NAME_DIR" "$PKG_META_DIR" "$DEDUPLICATED_PKGS"

PKGS_BY_TYPE="$DEST_DIR/by-type"
mkdir -pv "$PKGS_BY_TYPE"
"$SCRIPT_DIR/group-by-type.sh" "$DEDUPLICATED_PKGS" "$PKGS_BY_TYPE"

PKGS_BY_RELEVANCE="$DEST_DIR/by-relevance"
mkdir -pv "$PKGS_BY_RELEVANCE"
"$SCRIPT_DIR/group-by-relevance.sh" "$PKGS_BY_TYPE" "$PKGS_BY_RELEVANCE"

PKGS_BY_COMPARABILITY="$DEST_DIR/by-comparability"
mkdir -pv "$PKGS_BY_COMPARABILITY"
"$SCRIPT_DIR/group-by-comparability.sh" "$PKGS_BY_TYPE" "$PKGS_BY_COMPARABILITY"

