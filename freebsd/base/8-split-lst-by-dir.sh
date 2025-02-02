#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate split of all package listings by directory of origin

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    ALL_PKG_LISTING="$(realpath "$1")"
    SPLIT_LISTINGS_DIR="$(realpath "$2")"
else
    printf "Usage: %s [ALL_PKG_LISTING] [SPLIT_LISTINGS_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify ALL_PKG_LISTING and SPLIT_LISTINGS_DIR

if ! [ -r "$ALL_PKG_LISTING" ]; then
    printf "File \"%s\" is not readable or does not exist\n" "$ALL_PKG_LISTING"
    exit
fi

if ! [ -d "$SPLIT_LISTINGS_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$SPLIT_LISTINGS_DIR"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"

cd "$SPLIT_LISTINGS_DIR" || exit
awk -f "$WORK_DIR/generate-split-lst.awk" "$ALL_PKG_LISTING"
cd - || exit

usr_tmp="$(mktemp)"
cp "$SPLIT_LISTINGS_DIR/usr.lst" "$usr_tmp"

ex -s "$usr_tmp" << EOF
%s/^usr\///g
w
q
EOF

USR_SPLIT_LISTINGS_DIR="$SPLIT_LISTINGS_DIR/usr"
mkdir -pv "$USR_SPLIT_LISTINGS_DIR"
cd "$USR_SPLIT_LISTINGS_DIR" || exit
awk -f "$WORK_DIR/generate-split-lst.awk" "$usr_tmp"
cd - || exit

rm "$usr_tmp"

