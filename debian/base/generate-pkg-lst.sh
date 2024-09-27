#!/usr/bin/env awk

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Extract package file listings

{
    n = split($1, lines, "/")
    filename = lines[n]
    print $2 >> filename ".txt"
}

