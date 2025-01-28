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
    SPLIT_LISTING_DIR="$(realpath "$2")"
else
    printf "Usage: %s [ALL_PKG_LISTING] [SPLIT_LISTING_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify ALL_PKG_LISTING and SPLIT_LISTING_DIR

if ! [ -r "$ALL_PKG_LISTING" ]; then
    printf "File ALL_PKG_LISTING is not readable or does not exist\n"
    exit
fi

if ! [ -r "$SPLIT_LISTING_DIR" ]; then
    printf "File SPLIT_LISTING_DIR is not readable or does not exist\n"
    exit
fi

########################################

WORK_DIR="$(dirname "$(realpath "$0")")"

cd "$SPLIT_LISTING_DIR" || exit
awk -f "$WORK_DIR/generate-split-lst.awk" "$ALL_PKG_LISTING"
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
awk -f "$WORK_DIR/generate-split-lst.awk" "$usr_tmp"
cd - || exit

rm "$usr_tmp"

# Primary interest directories
# Binaries - /bin; /sbin; /usr/bin; /usr/sbin
# Libraries - /lib; /lib32; /lib64; /libx32; /usr/lib; /usr/lib32; /usr/lib64; /usr/libx32

