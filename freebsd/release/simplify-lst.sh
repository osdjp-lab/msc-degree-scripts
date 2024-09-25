#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Simplify freebsd package listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    BASE_LISTING="$(realpath "$1")"
    SIMPLIFIED_LISTING="$(realpath "$2")"
else
    printf "Usage: %s [BASE_LISTING] [SIMPLIFIED_LISTING]\n" "$(basename "$0")"
    exit
fi

# Verify BASE_LISTING and SIMPLIFIED_LISTING files

if ! [ -f "$BASE_LISTING" ]; then
    printf "Input file \"%s\" does not exist or is not a regular file\n" "$BASE_LISTING"
    exit
fi

if [ -f "$SIMPLIFIED_LISTING" ]; then
    printf "Output file \"%s\" exists\n" "$BASE_LISTING"
    exit
fi

########################################

cp -v "$BASE_LISTING" "$SIMPLIFIED_LISTING"

ex -s "$SIMPLIFIED_LISTING" << EOF
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

