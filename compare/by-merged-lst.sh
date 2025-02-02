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

ALL_COMMON_LST="$OUTPUT_DIR/all-common.lst"

cat "$OUTPUT_DIR/bin/common.lst" "$OUTPUT_DIR/lib/common.lst" \
    > "$ALL_COMMON_LST"

sort -o "$ALL_COMMON_LST" "$ALL_COMMON_LST"

# Inverse backtracking

FREEBSD_BIN="$FREEBSD_MERGED_LST_DIR/bin"
DEBIAN_BIN="$DEBIAN_MERGED_LST_DIR/bin"
FREEBSD_COMMON_BIN_FULL="$OUTPUT_DIR/bin/freebsd-common-full.lst"
DEBIAN_COMMON_BIN_FULL="$OUTPUT_DIR/bin/debian-common-full.lst"

true > "$FREEBSD_COMMON_BIN_FULL"
true > "$DEBIAN_COMMON_BIN_FULL"

while IFS= read -r file; do
    2>/dev/null grep -h "bin\/$file$" "$FREEBSD_BIN" \
        >> "$FREEBSD_COMMON_BIN_FULL"
    if [ "$?" -eq 2 ]; then
        2>/dev/null grep -Fh "$file" "$FREEBSD_BIN" \
            >> "$FREEBSD_COMMON_BIN_FULL"
    fi
    2>/dev/null grep -h "bin\/$file$" "$DEBIAN_BIN" \
        >> "$DEBIAN_COMMON_BIN_FULL"
    if [ "$?" -eq 2 ]; then
        2>/dev/null grep -Fh "$file" "$FREEBSD_BIN" \
            >> "$DEBIAN_COMMON_BIN_FULL"
    fi
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

FREEBSD_COMMON_FULL="$OUTPUT_DIR/freebsd-common-full.lst"

cat "$FREEBSD_COMMON_BIN_FULL" "$FREEBSD_COMMON_LIB_FULL" \
    > "$FREEBSD_COMMON_FULL"

sort -o "$FREEBSD_COMMON_FULL" "$FREEBSD_COMMON_FULL"

DEBIAN_COMMON_FULL="$OUTPUT_DIR/debian-common-full.lst"

cat "$DEBIAN_COMMON_BIN_FULL" "$DEBIAN_COMMON_LIB_FULL" \
    > "$DEBIAN_COMMON_FULL"

sort -o "$DEBIAN_COMMON_FULL" "$DEBIAN_COMMON_FULL"

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
    2>/dev/null grep -h "$file$" "$FREEBSD_PKG_CONTENTS_LST" \
        >> "$FREEBSD_COMMON_PKGS_CONTENTS"
    if [ "$?" -eq 2 ]; then
        2>/dev/null grep -Fh "$file" "$FREEBSD_PKG_CONTENTS_LST"\
            >> "$FREEBSD_COMMON_PKGS_CONTENTS"
    fi
done < "$OUTPUT_DIR/bin/freebsd-common-full.lst"

# Deduplicate overlap in search between sbin and usr/sbin as well as bin and usr/bin 

sort "$FREEBSD_COMMON_PKGS_CONTENTS" \
    | uniq | sponge "$FREEBSD_COMMON_PKGS_CONTENTS"

FREEBSD_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/lib/freebsd-common-pkgs-contents"

true > "$FREEBSD_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    2>/dev/null grep -h "$file$" "$FREEBSD_PKG_CONTENTS_LST" \
        >> "$FREEBSD_COMMON_PKGS_CONTENTS"
    if [ "$?" -eq 2 ]; then
        2>/dev/null grep -Fh "$file" "$FREEBSD_PKG_CONTENTS_LST" \
            >> "$FREEBSD_COMMON_PKGS_CONTENTS"
    fi
done < "$OUTPUT_DIR/lib/freebsd-common-full.lst"

sort "$FREEBSD_COMMON_PKGS_CONTENTS" \
    | uniq | sponge "$FREEBSD_COMMON_PKGS_CONTENTS"

DEBIAN_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/bin/debian-common-pkgs-contents"

true > "$DEBIAN_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    2>/dev/null grep -h "$file$" "$DEBIAN_PKG_CONTENTS_LST" \
        >> "$DEBIAN_COMMON_PKGS_CONTENTS"
    if [ "$?" -eq 2 ]; then
        2>/dev/null grep -Fh "$file" "$DEBIAN_PKG_CONTENTS_LST" \
            >> "$DEBIAN_COMMON_PKGS_CONTENTS"
    fi
done < "$OUTPUT_DIR/bin/debian-common-full.lst"

sort "$DEBIAN_COMMON_PKGS_CONTENTS" \
    | uniq | sponge "$DEBIAN_COMMON_PKGS_CONTENTS"

DEBIAN_COMMON_PKGS_CONTENTS="$OUTPUT_DIR/lib/debian-common-pkgs-contents"

true > "$DEBIAN_COMMON_PKGS_CONTENTS"

while IFS= read -r file; do
    2>/dev/null grep -h "$file$" "$DEBIAN_PKG_CONTENTS_LST" \
        >> "$DEBIAN_COMMON_PKGS_CONTENTS"
    if [ "$?" -eq 2 ]; then
        2>/dev/null grep -Fh "$file" "$DEBIAN_PKG_CONTENTS_LST" \
            >> "$DEBIAN_COMMON_PKGS_CONTENTS"
    fi
done < "$OUTPUT_DIR/lib/debian-common-full.lst"

sort "$DEBIAN_COMMON_PKGS_CONTENTS" \
    | uniq | sponge "$DEBIAN_COMMON_PKGS_CONTENTS"

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

# Match FreeBSD and Debian packages based on common files and exclude
# false positives

# Create listing containing 3 columns: freebsd-pkg, debian-pkg, file

FREEBSD_DEBIAN_PKG_LST="$OUTPUT_DIR/freebsd-debian-pkg.lst"
true > "$FREEBSD_DEBIAN_PKG_LST"

printf "file\tfreebsd-pkg\tdebian-pkg\n" >> "$FREEBSD_DEBIAN_PKG_LST"
printf "\n" >> "$FREEBSD_DEBIAN_PKG_LST"

EXCEPTIONS="$OUTPUT_DIR/freebsd-debian-exceptions"
mkdir "$EXCEPTIONS"

while IFS= read -r file; do

    # Get FreeBSD match count and string
    FREEBSD_MATCH_COUNT=$(2>/dev/null grep -c "\/$file$" \
        "$FREEBSD_ALL_COMMON_PKGS_CONTENTS")
    freebsd_pkg=$(2>/dev/null grep -h "\/$file$" \
        "$FREEBSD_ALL_COMMON_PKGS_CONTENTS" \
        | sed -e 's/^\(.*\)\t\(.*\)$/\1/')
    if [ "$FREEBSD_MATCH_COUNT" = "" ]; then
        FREEBSD_MATCH_COUNT=$(2>/dev/null grep -Fc "$file" \
            "$FREEBSD_ALL_COMMON_PKGS_CONTENTS")
        freebsd_pkg=$(2>/dev/null grep -Fh "$file" \
            "$FREEBSD_ALL_COMMON_PKGS_CONTENTS" \
            | sed -e 's/^\(.*\)\t\(.*\)$/\1/')
    fi
    
    # Get Debian match count and string
    DEBIAN_MATCH_COUNT=$(2>/dev/null grep -c "\/$file$" \
        "$DEBIAN_ALL_COMMON_PKGS_CONTENTS")
    debian_pkg=$(2>/dev/null grep -h "\/$file$" \
        "$DEBIAN_ALL_COMMON_PKGS_CONTENTS" \
        | sed -e 's/^\(.*\)\t\(.*\)$/\1/')
    if [ "$DEBIAN_MATCH_COUNT" = "" ]; then
        DEBIAN_MATCH_COUNT=$(2>/dev/null grep -Fc "$file" \
            "$DEBIAN_ALL_COMMON_PKGS_CONTENTS")
        debian_pkg=$(2>/dev/null grep -Fh "$file" \
            "$DEBIAN_ALL_COMMON_PKGS_CONTENTS" \
            | sed -e 's/^\(.*\)\t\(.*\)$/\1/')
    fi

    if [ "$FREEBSD_MATCH_COUNT" -eq 1 ]; then
        if [ "$DEBIAN_MATCH_COUNT" -eq 1 ]; then
            # 1:1
            printf "%s\t%s\t%s\n" "$file" "${freebsd_pkg}" \
                "${debian_pkg}" >> "$FREEBSD_DEBIAN_PKG_LST"
        else
            # 1:n
            printf "freebsd-pkg:\n\n" > "$EXCEPTIONS/$file"
            printf "%s\n" "${freebsd_pkg}" >> "$EXCEPTIONS/$file"
            printf "\ndebian-pkgs:\n\n" >> "$EXCEPTIONS/$file"
            printf "%s\n" "${debian_pkg}" >> "$EXCEPTIONS/$file"
        fi
    else
        if [ "$DEBIAN_MATCH_COUNT" -eq 1 ]; then
            # m:1
            printf "freebsd-pkgs:\n\n" > "$EXCEPTIONS/$file"
            printf "%s\n" "${freebsd_pkg}" >> "$EXCEPTIONS/$file"
            printf "\ndebian-pkg:\n\n" >> "$EXCEPTIONS/$file"
            printf "%s\n" "${debian_pkg}" >> "$EXCEPTIONS/$file"
        else
            # m:n
            printf "freebsd-pkgs:\n\n" > "$EXCEPTIONS/$file"
            printf "%s\n" "${freebsd_pkg}" >> "$EXCEPTIONS/$file"
            printf "\ndebian-pkgs:\n\n" >> "$EXCEPTIONS/$file"
            printf "%s\n" "${debian_pkg}" >> "$EXCEPTIONS/$file"
        fi

    fi
done < "$ALL_COMMON_LST"

column -t "$FREEBSD_DEBIAN_PKG_LST" | sort -k2,3 \
    | sponge "$FREEBSD_DEBIAN_PKG_LST"

