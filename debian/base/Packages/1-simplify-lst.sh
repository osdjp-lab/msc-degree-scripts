#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Simplify debian package listing

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_LISTING="$(realpath "$1")"
    SIMPLIFIED_LISTING="$(realpath "$2")"
else
    printf "Usage: %s [PKG_LISTING] [SIMPLIFIED_LISTING]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_LISTING and SIMPLIFIED_LISTING files

if ! [ -f "$PKG_LISTING" ]; then
    printf "Input file \"%s\" does not exist or is not a regular file\n" "$PKG_LISTING"
    exit
fi

if [ -f "$SIMPLIFIED_LISTING" ]; then
    printf "Output file \"%s\" exists\n" "$SIMPLIFIED_LISTING"
    exit
fi

########################################

cp -v "$PKG_LISTING" "$SIMPLIFIED_LISTING"

ex -s "$SIMPLIFIED_LISTING" << EOF
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

