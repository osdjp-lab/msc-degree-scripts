#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group packages by priority
# Priority: required > important > standard > optional > extra

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    PKG_META_DIR="$(realpath "$1")"
    PKGS_BY_PRIORITY_DIR="$(realpath "$2")"
else
    printf "Usage: %s [PKG_META_DIR] [PKGS_BY_PRIORITY_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify PKG_META_DIR and PKGS_BY_PRIORITY

if ! [ -d "$PKG_META_DIR" ]; then
    printf "PKG_META_DIR does not exist or is not a directory\n"
    exit
fi

if ! [ -d "$PKGS_BY_PRIORITY_DIR" ]; then
    printf "PKGS_BY_PRIORITY_DIR does not exist or is not a directory\n"
    exit
fi

########################################

TARGET_DIR="$PKGS_BY_PRIORITY_DIR/required"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: required" "$PKG_META_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$PKGS_BY_PRIORITY_DIR/important"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: important" "$PKG_META_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$PKGS_BY_PRIORITY_DIR/standard"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: standard" "$PKG_META_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$PKGS_BY_PRIORITY_DIR/optional"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: optional" "$PKG_META_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 

TARGET_DIR="$PKGS_BY_PRIORITY_DIR/extra"
mkdir -pv "$TARGET_DIR"

grep -ril "Priority: extra" "$PKG_META_DIR" | while IFS= read -r pkg
do
    ln -vsr "$pkg" "$TARGET_DIR"
done 


