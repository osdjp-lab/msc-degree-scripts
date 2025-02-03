#!/usr/bin/env awk

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Split category prefixed multi package entries into individual
# entries for the same file in Contents-amd64

/,/ {
    split($2, arr, ",")
    for (i in arr) {
        sub(/^[a-z]*\//, "", arr[i])
        print $1, arr[i]
    }
}

!/,/ {
    print $0
}

