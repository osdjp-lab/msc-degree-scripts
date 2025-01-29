#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract FreeBSD-14.1-base-release-1 package metadata
# from "packagesite.yaml" file and individual .pkg files

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    DEST_DIR="$(realpath "$1")"
else
    printf "Usage: %s [DEST_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify value of DEST_DIR

if ! [ -d "$DEST_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$DEST_DIR"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"
SCRIPT_DIR="$WORK_DIR/base"

SRC_DIR="$DEST_DIR/src"
if ! [ -d "$SRC_DIR" ]; then
    mkdir -pv "$SRC_DIR"
    "$SCRIPT_DIR/1-get-data.sh" "$SRC_DIR"
fi

BASE_LISTING="$SRC_DIR/packagesite.yaml" 
SIMPLIFIED_LISTING="$DEST_DIR/simplified"
"$SCRIPT_DIR/2-simplify-lst.sh" "$BASE_LISTING" "$SIMPLIFIED_LISTING"

PKG_META_DIR="$DEST_DIR/pkgs"
mkdir -pv "$PKG_META_DIR"
"$SCRIPT_DIR/3-split-lst.sh" "$SIMPLIFIED_LISTING" "$PKG_META_DIR"

PKG_FILES_DIR="$SRC_DIR/pkgs"
LISTING_DIR="$DEST_DIR/pkg-listings"
mkdir -pv "$LISTING_DIR"
"$SCRIPT_DIR/4-generate-pkg-lst.sh" "$PKG_FILES_DIR" "$LISTING_DIR"

SIMPLIFIED_LISTING_DIR="$DEST_DIR/deduplicated-simplified-pkg-listings"
mkdir -pv "$SIMPLIFIED_LISTING_DIR"
"$SCRIPT_DIR/5-generate-simplified-pkg-lst.sh" "$LISTING_DIR" "$SIMPLIFIED_LISTING_DIR"

NAME_DIR="$DEST_DIR/names"
mkdir -pv "$NAME_DIR"
ALL_PKG_LISTING="$NAME_DIR/all-pkg.lst"
"$SCRIPT_DIR/6-generate-all-pkg-lst.sh" "$LISTING_DIR" "$ALL_PKG_LISTING"

SPLIT_LISTINGS_DIR="$DEST_DIR/split-lst"
mkdir -pv "$SPLIT_LISTINGS_DIR"
"$SCRIPT_DIR/7-split-lst-by-dir.sh" "$ALL_PKG_LISTING" "$SPLIT_LISTINGS_DIR"

MERGED_LISTINGS_DIR="$DEST_DIR/merged-lst"
mkdir -pv "$MERGED_LISTINGS_DIR"
"$SCRIPT_DIR/8-generate-merged-dir-lst.sh" "$SPLIT_LISTINGS_DIR" "$MERGED_LISTINGS_DIR"

MAN_DIR="$DEST_DIR/pkg-man-pages"
mkdir -pv "$MAN_DIR"
"$SCRIPT_DIR/9-extract-man-pages.sh" "$PKG_FILES_DIR" "$MAN_DIR"
"$SCRIPT_DIR/10-deduplicate-man-pages.sh" "$MAN_DIR"

PLAIN_MAN_DIR="$DEST_DIR/pkg-man-txt"
mkdir -pv "$PLAIN_MAN_DIR"
"$SCRIPT_DIR/11-extract-txt-man-pages.sh" "$MAN_DIR" "$PLAIN_MAN_DIR"
"$SCRIPT_DIR/12-group-txt-man-pages.sh" "$PLAIN_MAN_DIR"

"$SCRIPT_DIR/13-generate-names.sh" "$NAME_DIR" "$PKG_META_DIR" "$MAN_DIR"
"$SCRIPT_DIR/14-generate-pkg-contents-lst.sh" "$SIMPLIFIED_LISTING_DIR" "$NAME_DIR"

DEDUPLICATED_PKGS="$DEST_DIR/pkgs-no-man-dbg-dev-lib32"
mkdir -pv "$DEDUPLICATED_PKGS"
"$SCRIPT_DIR/15-deduplicate-pkgs.sh" "$NAME_DIR" "$PKG_META_DIR" "$DEDUPLICATED_PKGS"

PKGS_BY_TYPE="$DEST_DIR/by-type"
mkdir -pv "$PKGS_BY_TYPE"
"$SCRIPT_DIR/16-group-by-type.sh" "$DEDUPLICATED_PKGS" "$PKGS_BY_TYPE"

PKGS_BY_RELEVANCE="$DEST_DIR/by-relevance"
mkdir -pv "$PKGS_BY_RELEVANCE"
"$SCRIPT_DIR/17-group-by-relevance.sh" "$PKGS_BY_TYPE" "$PKGS_BY_RELEVANCE"

PKGS_BY_COMPARABILITY="$DEST_DIR/by-comparability"
mkdir -pv "$PKGS_BY_COMPARABILITY"
"$SCRIPT_DIR/18-group-by-comparability.sh" "$PKGS_BY_TYPE" "$PKGS_BY_COMPARABILITY"

