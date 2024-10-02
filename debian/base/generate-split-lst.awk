#!/usr/bin/env awk

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Generate all package contents split listings by directory of origin

BEGIN {
    FS = "/"
}

$1 != field {
    if (field != "")
        close(field ".lst")
    field = $1
}

$1 == field {
    print $0 >> field ".lst"
}

