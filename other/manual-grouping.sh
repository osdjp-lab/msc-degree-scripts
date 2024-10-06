#/usr/bin/env sh

set -x

# Manually sort debian package meta files based on freebsd base package files

FREEBSD_BASE_LISTING="../../freebsd-14.1-base/names/no-man-dbg-dev-lib32"
FREEBSD_BASE_PKGS="../../freebsd-14.1-base/pkgs"
DEBIAN_PKGS="../../debian-bookworm-base-main/pkgs"

# Make directories for all packages

MATCH_FAIL="no-match-pkgs"

mkdir -pv "$MATCH_FAIL"

while read packagename; do
    # Allow selection of corresponding debian package metadata files
    pushd $DEBIAN_PKGS
    SELECTED_PKGS=$(find . -type f | fzf -e -m --query "$packagename")
    RETURN_VALUE=$?
    popd
    if [ $RETURN_VALUE -eq 130 ]; then
        cp -vt "$MATCH_FAIL" "$FREEBSD_BASE_PKGS/FreeBSD-$packagename.txt"
        continue
    fi
    # Make package directory
    mkdir -pv "$packagename"
    # Copy freebsd package meta file to directory
    cp -vt "$packagename/" "$FREEBSD_BASE_PKGS/FreeBSD-$packagename.txt"
    if [ $(printf "%s\n" $SELECTED_PKGS | wc -l) -eq 1 ]; then
        TARGET_DIR=$(realpath "$packagename")
        pushd $DEBIAN_PKGS
        cp -vt "$TARGET_DIR" "$SELECTED_PKGS"
        popd
    else
        # Make debian subdirectory
        mkdir -pv "$packagename/debian"
        # Copy selected packages
        TARGET_DIR=$(realpath "$packagename/debian")
        pushd $DEBIAN_PKGS
        cp -vt "$TARGET_DIR" "$SELECTED_PKGS"
        popd
    fi
done < $FREEBSD_BASE_LISTING

