#!/usr/bin/env sh

set -x

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Manual grouping of FreeBSD base system packages by type

# Verify number of positional parameters

if [ $# -eq 1 ]; then
    WORK_DIR="$(realpath "$1")"
else
    printf "Usage: %s [WORK_DIR]\n" "$(basename "$0")"
    exit
fi

# Verify value of WORK_DIR

if ! [ -d "$WORK_DIR" ]; then
    printf "Work directory \"%s\" does not exist or is not a directory\n" "$WORK_DIR"
    exit
fi

FREEBSD_BASE_LISTING="$WORK_DIR/names/no-man-dbg-dev-lib32"
FREEBSD_BASE_PKGS="$WORK_DIR/pkgs-no-man-dbg-dev-lib32"

# Check if "$FREEBSD_BASE_LISTING" file is present and readable in WORK_DIR

if ! [ -r "$FREEBSD_BASE_LISTING" ]; then
    printf "File \"names/no-man-dbg-dev-lib32\" in work directory is not readable or does not exist\n" 
    exit
fi

# Check if "$FREEBSD_BASE_PKGS" directory is present and readable in WORK_DIR

if ! [ -d "$FREEBSD_BASE_PKGS" ]; then
    printf "\"pkgs-no-man-dbg-dev-lib32\" does not exist or is not a directory\n"
    exit
fi

BY_TYPE="$WORK_DIR/by-type"

mkdir -pv "$BY_TYPE"

# Make copy of FreeBSD base packages

BASE="$BY_TYPE/base"
mkdir -pv "$BASE"

find "$FREEBSD_BASE_PKGS" -exec cp -vt "$BASE" {} +

ls "$BASE"

# Libraries

DEST_DIR="$BY_TYPE/lib"
mkdir -pv "$DEST_DIR"

KEY="lib"
find "$BASE" -name "*$KEY*" -exec mv -vt "$DEST_DIR" {} +

# Packet filtering

DEST_DIR="$BY_TYPE/net/pf"
mkdir -pv "$DEST_DIR"

KEY="pf"
grep -lri "$KEY" "$BASE" | xargs mv -vt "$DEST_DIR"
mv -vt "$DEST_DIR" "$BASE/FreeBSD-blocklist.txt"

# Kernel images

DEST_DIR="$BY_TYPE/system/kernel"
mkdir -pv "$DEST_DIR"

KEY="kernel"
find "$BASE" -name "*$KEY*" -exec mv -vt "$DEST_DIR" {} +

# Editors

DEST_DIR="$BY_TYPE/system/editor"
mkdir -pv "$DEST_DIR"

KEY="editor"
grep -lri "$KEY" "$BASE" | xargs mv -vt "$DEST_DIR"

# Dev-tools

DEST_DIR="$BY_TYPE/dev-tools"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-clang.txt" \
                   "$BASE/FreeBSD-lldb.txt" \
                   "$BASE/FreeBSD-elftoolchain.txt" \
                   "$BASE/FreeBSD-tests.txt" \
                   "$BASE/FreeBSD-ctf-tools.txt" \
                   "$BASE/FreeBSD-dtrace.txt" \
                   "$BASE/FreeBSD-dwatch.txt" \
                   "$BASE/FreeBSD-lld.txt"

# SSL

DEST_DIR="$BY_TYPE/net/ssl"
mkdir -pv "$DEST_DIR"

KEY="ssl"
grep -lri "$KEY" "$BASE" | xargs mv -vt "$DEST_DIR"

# Other

DEST_DIR="$BY_TYPE/other"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-games.txt" \
                   "$BASE/FreeBSD-src.txt" \
                   "$BASE/FreeBSD-bsdinstall.txt" \
                   "$BASE/FreeBSD-pkg-bootstrap.txt" \
                   "$BASE/FreeBSD-src-sys.txt"

# Power management

DEST_DIR="$BY_TYPE/other/power"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-acpi.txt" \
                   "$BASE/FreeBSD-apm.txt"

# System

DEST_DIR="$BY_TYPE/system"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-runtime.txt" \
                   "$BASE/FreeBSD-csh.txt" \
                   "$BASE/FreeBSD-devd.txt" \
                   "$BASE/FreeBSD-rescue.txt" \
                   "$BASE/FreeBSD-devmatch.txt" \
                   "$BASE/FreeBSD-acct.txt" \
                   "$BASE/FreeBSD-bootloader.txt" \
                   "$BASE/FreeBSD-syslogd.txt" \
                   "$BASE/FreeBSD-efi-tools.txt" \
                   "$BASE/FreeBSD-rc.txt"

# Networking

DEST_DIR="$BY_TYPE/net"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-dhclient.txt" \
                   "$BASE/FreeBSD-telnet.txt" \
                   "$BASE/FreeBSD-nuageinit.txt" \
                   "$BASE/FreeBSD-ppp.txt" \
                   "$BASE/FreeBSD-yp.txt" \
                   "$BASE/FreeBSD-tcpd.txt" \
                   "$BASE/FreeBSD-ftp.txt" \
                   "$BASE/FreeBSD-fetch.txt" \
                   "$BASE/FreeBSD-ssh.txt" \
                   "$BASE/FreeBSD-rcmds.txt" \
                   "$BASE/FreeBSD-kerberos.txt" \
                   "$BASE/FreeBSD-bluetooth.txt" \
                   "$BASE/FreeBSD-netmap.txt" \
                   "$BASE/FreeBSD-ftpd.txt" \
                   "$BASE/FreeBSD-natd.txt" \
                   "$BASE/FreeBSD-inetd.txt"

# Email

DEST_DIR="$BY_TYPE/net/mail"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-dma.txt" \
                   "$BASE/FreeBSD-sendmail.txt"

# WiFi 802.11

DEST_DIR="$BY_TYPE/net/wifi"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-wpa.txt" \
                   "$BASE/FreeBSD-hostapd.txt"

# Networking file systems

DEST_DIR="$BY_TYPE/net/fs"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-nfs.txt" \
                   "$BASE/FreeBSD-hast.txt" \
                   "$BASE/FreeBSD-geom.txt" \
                   "$BASE/FreeBSD-ggate.txt"

# DNS

DEST_DIR="$BY_TYPE/net/dns"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-unbound.txt" \
                   "$BASE/FreeBSD-resolvconf.txt"

# Virtualization

DEST_DIR="$BY_TYPE/virt"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-bhyve.txt" \
                   "$BASE/FreeBSD-jail.txt" \
                   "$BASE/FreeBSD-hyperv-tools.txt"
# Drivers

DEST_DIR="$BY_TYPE/driver"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-syscons.txt" \
                   "$BASE/FreeBSD-smbutils.txt" \
                   "$BASE/FreeBSD-rdma.txt" \
                   "$BASE/FreeBSD-nvme-tools.txt" \
                   "$BASE/FreeBSD-mlx-tools.txt" \
                   "$BASE/FreeBSD-cxgbe-tools.txt" \
                   "$BASE/FreeBSD-bsnmp.txt" \
                   "$BASE/FreeBSD-ccdconfig.txt" \
                   "$BASE/FreeBSD-autofs.txt" \
                   "$BASE/FreeBSD-iscsi.txt"

# Scheduling

DEST_DIR="$BY_TYPE/util/scheduling"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-at.txt" \
                   "$BASE/FreeBSD-periodic.txt"

# Utilities

DEST_DIR="$BY_TYPE/util"
mkdir -pv "$DEST_DIR"

KEY="util"
grep -lri "$KEY" "$BASE" | xargs mv -vt "$DEST_DIR"
mv -vt "$DEST_DIR" "$BASE/FreeBSD-dpv.txt" \
                   "$BASE/FreeBSD-ufs.txt" \
                   "$BASE/FreeBSD-zfs.txt" \
                   "$BASE/FreeBSD-quotacheck.txt"

# Data

DEST_DIR="$BY_TYPE/data"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-vt-data.txt" \
                   "$BASE/FreeBSD-zoneinfo.txt" \
                   "$BASE/FreeBSD-mtree.txt" \
                   "$BASE/FreeBSD-locales.txt"

rmdir -v "$BASE"

