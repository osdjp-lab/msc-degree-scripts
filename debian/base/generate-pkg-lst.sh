#!/usr/bin/env awk

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract package file listings

$1 != field1 {
    if (filename != "")
        close(filename ".txt")
    field1 = $1
    n = split($1, lines, "/")
    filename = lines[n]
}

$1 == field1 {
    print $2 >> filename ".txt"
}

