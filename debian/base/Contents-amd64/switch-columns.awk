#!/usr/bin/env awk

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Switch columns 1 and 2 in Contents-amd64

# Moving second field to first place because it is the only field
# guaranteed to not have any spaces
# Printing remaining field as they all belong to the file path

{
    printf ("%s ", $NF)
    for (i = 1; i < NF; i++) {
        printf ("%s ", $i)
    }
    printf ("\n")
}

