#!/usr/bin/env awk

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Split category prefixed multi package entries into individual
# entries for the same file in Contents-amd64

$1 ~ /,/ {
    split($1, arr, ",")
    for (i in arr) {
        sub(/^.*\//, "", arr[i])
        printf ("%s ", arr[i])
        for (i = 2; i <= NF; i++) {
            printf ("%s ", $i)
        }
        printf ("\n")
    }
}

$1 !~ /,/ {
    sub(/^.*\//, "", $1)
    print $0
}

