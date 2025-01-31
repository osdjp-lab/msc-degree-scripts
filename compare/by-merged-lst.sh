#!/usr/bin/env sh

set -x

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Find complementary Debian and FreeBSD packages based on merged
# listings for bin (binaries) and lib (libraries)

# Verify number of positional parameters

if [ $# -eq 5 ]; then
    FREEBSD_MERGED_LST_DIR="$(realpath "$1")"
    DEBIAN_MERGED_LST_DIR="$(realpath "$2")"
    FREEBSD_PKG_CONTENTS_LST="$(realpath "$3")"
    DEBIAN_PKG_CONTENTS_LST="$(realpath "$4")"
    OUTPUT_DIR="$(realpath "$5")"
else
    printf "Usage: %s [FREEBSD_MERGED_LST_DIR] [DEBIAN_MERGED_LST_DIR] [FREEBSD_PKG_CONTENTS_LST] [DEBIAN_PKG_CONTENTS_LST] [OUTPUT_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify arguments

if ! [ -d "$FREEBSD_MERGED_LST_DIR" ]; then
    printf "Source \"%s\" does not exist or is not a directory\n" "$FREEBSD_MERGED_LST_DIR"
    exit
fi

if ! [ -d "$DEBIAN_MERGED_LST_DIR" ]; then
    printf "Source \"%s\" does not exist or is not a directory\n" "$DEBIAN_MERGED_LST_DIR"
    exit
fi

if ! [ -r "$FREEBSD_PKG_CONTENTS_LST" ]; then
    printf "File \"%s\" in source directory is not readable or does not exist\n" "$FREEBSD_PKG_CONTENTS_LST"
    exit
fi

if ! [ -r "$DEBIAN_PKG_CONTENTS_LST" ]; then
    printf "File \"%s\" in source directory is not readable or does not exist\n" "$DEBIAN_PKG_CONTENTS_LST"
    exit
fi

if ! [ -d "$OUTPUT_DIR" ]; then
    printf "Destination \"%s\" does not exist or is not a directory\n" "$OUTPUT_DIR"
    exit
fi

########################################

# Comparison

FREEBSD_BIN="$FREEBSD_MERGED_LST_DIR/bin-deduplicated"
FREEBSD_LIB="$FREEBSD_MERGED_LST_DIR/lib-deduplicated"
DEBIAN_BIN="$DEBIAN_MERGED_LST_DIR/bin-deduplicated"
DEBIAN_LIB="$DEBIAN_MERGED_LST_DIR/lib-deduplicated"

TARGET_DIR="$OUTPUT_DIR/bin"
mkdir -pv "$TARGET_DIR"
comm -12 "$FREEBSD_BIN" "$DEBIAN_BIN" > "$TARGET_DIR/common.lst"
comm -23 "$FREEBSD_BIN" "$DEBIAN_BIN" > "$TARGET_DIR/freebsd-only.lst"
comm -13 "$FREEBSD_BIN" "$DEBIAN_BIN" > "$TARGET_DIR/debian-only.lst"

TARGET_DIR="$OUTPUT_DIR/lib"
mkdir -pv "$TARGET_DIR"
comm -12 "$FREEBSD_LIB" "$DEBIAN_LIB" > "$TARGET_DIR/common.lst"
comm -23 "$FREEBSD_LIB" "$DEBIAN_LIB" > "$TARGET_DIR/freebsd-only.lst"
comm -13 "$FREEBSD_LIB" "$DEBIAN_LIB" > "$TARGET_DIR/debian-only.lst"

# Inverse backtracking

FREEBSD_BIN="$FREEBSD_MERGED_LST_DIR/bin"
DEBIAN_BIN="$DEBIAN_MERGED_LST_DIR/bin"
FREEBSD_COMMON_BIN_FULL="$OUTPUT_DIR/bin/freebsd-common-full.lst"
DEBIAN_COMMON_BIN_FULL="$OUTPUT_DIR/bin/debian-common-full.lst"

true > "$FREEBSD_COMMON_BIN_FULL"
true > "$DEBIAN_COMMON_BIN_FULL"

while IFS= read -r file; do
    grep -Fhm 1 "$file" "$FREEBSD_BIN" >> "$FREEBSD_COMMON_BIN_FULL"
    grep -Fhm 1 "$file" "$DEBIAN_BIN" >> "$DEBIAN_COMMON_BIN_FULL"
done < "$OUTPUT_DIR/bin/common.lst"

FREEBSD_LIB="$FREEBSD_MERGED_LST_DIR/lib"
DEBIAN_LIB="$DEBIAN_MERGED_LST_DIR/lib"
FREEBSD_COMMON_LIB_FULL="$OUTPUT_DIR/lib/freebsd-common-full.lst"
DEBIAN_COMMON_LIB_FULL="$OUTPUT_DIR/lib/debian-common-full.lst"

true > "$FREEBSD_COMMON_LIB_FULL"
true > "$DEBIAN_COMMON_LIB_FULL"

while IFS= read -r file; do
    grep -Fhm 1 "$file" "$FREEBSD_LIB" >> "$FREEBSD_COMMON_LIB_FULL"
    grep -Fhm 1 "$file" "$DEBIAN_LIB" >> "$DEBIAN_COMMON_LIB_FULL"
done < "$OUTPUT_DIR/lib/common.lst"

# Cross-reference (package-contents listings)

# 1. Package names for full listings
# 2. Number of files in package vs number of matched (%)

# Types of package matches: 1:n; n:1
# Exceptions n=0 (no matches) and n=1 (one match)

# Types of listing matches:
# None, Partial, Full

FREEBSD_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/bin/freebsd-common-pkgs-contents"

true > "$FREEBSD_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    grep -Fhm 1 "$file" "$FREEBSD_PKG_CONTENTS_LST" >> "$FREEBSD_COMMON_PKGS_CONTENTS"
done < "$OUTPUT_DIR/bin/freebsd-common-full.lst"

FREEBSD_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/lib/freebsd-common-pkgs-contents"

true > "$FREEBSD_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    grep -Fhm 1 "$file" "$FREEBSD_PKG_CONTENTS_LST" >> "$FREEBSD_COMMON_PKGS_CONTENTS"
done < "$OUTPUT_DIR/lib/freebsd-common-full.lst"

DEBIAN_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/bin/debian-common-pkgs-contents"

true > "$DEBIAN_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    grep -Fhm 1 "$file" "$DEBIAN_PKG_CONTENTS_LST" >> "$DEBIAN_COMMON_PKGS_CONTENTS"
done < "$OUTPUT_DIR/bin/debian-common-full.lst"

DEBIAN_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/lib/debian-common-pkgs-contents"

true > "$DEBIAN_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    grep -Fhm 1 "$file" "$DEBIAN_PKG_CONTENTS_LST" >> "$DEBIAN_COMMON_PKGS_CONTENTS"
done < "$OUTPUT_DIR/lib/debian-common-full.lst"

# Cross-reference (packages only)

DEBIAN_ALL_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/debian-all-common-pkgs-contents"

cat "$OUTPUT_DIR/bin/debian-common-pkgs-contents" \
    "$OUTPUT_DIR/lib/debian-common-pkgs-contents" \
    > "$DEBIAN_ALL_COMMON_PKGS_CONTENTS"

sort -o "$DEBIAN_ALL_COMMON_PKGS_CONTENTS" \
        "$DEBIAN_ALL_COMMON_PKGS_CONTENTS"

DEBIAN_ALL_COMMON_PKGS="$OUTPUT_DIR/debian-all-common-pkgs"

cut -f1 "$DEBIAN_ALL_COMMON_PKGS_CONTENTS" > "$DEBIAN_ALL_COMMON_PKGS"

uniq "$DEBIAN_ALL_COMMON_PKGS" | sponge "$DEBIAN_ALL_COMMON_PKGS"

FREEBSD_ALL_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/freebsd-all-common-pkgs-contents"

cat "$OUTPUT_DIR/bin/freebsd-common-pkgs-contents" \
    "$OUTPUT_DIR/lib/freebsd-common-pkgs-contents" \
    > "$FREEBSD_ALL_COMMON_PKGS_CONTENTS"

sort -o "$FREEBSD_ALL_COMMON_PKGS_CONTENTS" \
        "$FREEBSD_ALL_COMMON_PKGS_CONTENTS"

FREEBSD_ALL_COMMON_PKGS="$OUTPUT_DIR/freebsd-all-common-pkgs"

cut -f1 "$FREEBSD_ALL_COMMON_PKGS_CONTENTS" > "$FREEBSD_ALL_COMMON_PKGS"

uniq "$FREEBSD_ALL_COMMON_PKGS" | sponge "$FREEBSD_ALL_COMMON_PKGS"

# Exclude false positive matches

# Create listing containing 3 columns: freebsd-pkg, debian-pkg, file

