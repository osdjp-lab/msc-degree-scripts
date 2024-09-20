#!/usr/bin/env sh

#
# SPDX-License-Identifier: MIT
# 
# Copyright (C) 2024 Oscar Szumiak
#

# Group packages by type

# Verify number of positional parameters

if [ $# -eq 2 ]; then
    DEDUPLICATED_PKGS="$(realpath "$1")"
    PKGS_BY_TYPE="$(realpath "$2")"
else
    printf "Usage: %s [DEDUPLICATED_PKGS] [PKGS_BY_TYPE]\n" "$(basename "$0")"
    exit
fi

# Verify DEDUPLICATED_PKGS and PKGS_BY_TYPE

if ! [ -d "$DEDUPLICATED_PKGS" ]; then
    printf "Deduplicated pkgs directory \"%s\" does not exist or is not a directory\n" "$DEDUPLICATED_PKGS"
    exit
fi

if ! [ -d "$PKGS_BY_TYPE" ]; then
    printf "Pkgs by type directory \"%s\" does not exist or is not a directory\n" "$PKGS_BY_TYPE"
    exit
fi

########################################

# Make copy of FreeBSD base packages

BASE="$PKGS_BY_TYPE/base"
mkdir -pv "$BASE"

find "$DEDUPLICATED_PKGS" -exec cp -vt "$BASE" {} +

# Bundles

DEST_DIR="$PKGS_BY_TYPE/bundle"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-runtime.txt" \
                   "$BASE/FreeBSD-utilities.txt"

# Data

DEST_DIR="$PKGS_BY_TYPE/data"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-locales.txt" \
                   "$BASE/FreeBSD-mtree.txt" \
                   "$BASE/FreeBSD-syscons.txt" \
                   "$BASE/FreeBSD-vt-data.txt" \
                   "$BASE/FreeBSD-zoneinfo.txt"

# Dev-tools

DEST_DIR="$PKGS_BY_TYPE/dev-tools"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-clang.txt" \
                   "$BASE/FreeBSD-ctf-tools.txt" \
                   "$BASE/FreeBSD-dtrace.txt" \
                   "$BASE/FreeBSD-dwatch.txt" \
                   "$BASE/FreeBSD-elftoolchain.txt" \
                   "$BASE/FreeBSD-lldb.txt" \
                   "$BASE/FreeBSD-lld.txt" \
                   "$BASE/FreeBSD-tests.txt"

# Libraries

DEST_DIR="$PKGS_BY_TYPE/lib"
mkdir -pv "$DEST_DIR"

KEY="lib"
find "$BASE" -name "*$KEY*" -exec mv -vt "$DEST_DIR" {} +

# Networking

DEST_DIR="$PKGS_BY_TYPE/net"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-bluetooth.txt" \
                   "$BASE/FreeBSD-bsnmp.txt" \
                   "$BASE/FreeBSD-dhclient.txt" \
                   "$BASE/FreeBSD-fetch.txt" \
                   "$BASE/FreeBSD-ftpd.txt" \
                   "$BASE/FreeBSD-ftp.txt" \
                   "$BASE/FreeBSD-inetd.txt" \
                   "$BASE/FreeBSD-kerberos.txt" \
                   "$BASE/FreeBSD-natd.txt" \
                   "$BASE/FreeBSD-netmap.txt" \
                   "$BASE/FreeBSD-nuageinit.txt" \
                   "$BASE/FreeBSD-ppp.txt" \
                   "$BASE/FreeBSD-rcmds.txt" \
                   "$BASE/FreeBSD-rdma.txt" \
                   "$BASE/FreeBSD-ssh.txt" \
                   "$BASE/FreeBSD-tcpd.txt" \
                   "$BASE/FreeBSD-telnet.txt" \
                   "$BASE/FreeBSD-yp.txt"

## DNS

DEST_DIR="$PKGS_BY_TYPE/net/dns"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-resolvconf.txt" \
                   "$BASE/FreeBSD-unbound.txt"

## Ethernet adapter drivers

DEST_DIR="$PKGS_BY_TYPE/driver"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-cxgbe-tools.txt" \
                   "$BASE/FreeBSD-mlx-tools.txt"

## Email

DEST_DIR="$PKGS_BY_TYPE/net/mail"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-dma.txt" \
                   "$BASE/FreeBSD-sendmail.txt"

## Packet filtering

DEST_DIR="$PKGS_BY_TYPE/net/packet-filter"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-blocklist.txt" \
                   "$BASE/FreeBSD-ipfilter.txt" \
                   "$BASE/FreeBSD-ipf.txt" \
                   "$BASE/FreeBSD-ipfw.txt" \
                   "$BASE/FreeBSD-pf.txt"

## SSL

DEST_DIR="$PKGS_BY_TYPE/net/ssl"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-caroot.txt" \
                   "$BASE/FreeBSD-certctl.txt" \
                   "$BASE/FreeBSD-openssl.txt"

## Network storage

DEST_DIR="$PKGS_BY_TYPE/net/storage"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-ggate.txt" \
                   "$BASE/FreeBSD-hast.txt" \
                   "$BASE/FreeBSD-iscsi.txt" \
                   "$BASE/FreeBSD-nfs.txt"

## WiFi 802.11

DEST_DIR="$PKGS_BY_TYPE/net/wifi"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-hostapd.txt" \
                   "$BASE/FreeBSD-wpa.txt"

# Other

DEST_DIR="$PKGS_BY_TYPE/other"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-bsdinstall.txt" \
                   "$BASE/FreeBSD-fwget.txt" \
                   "$BASE/FreeBSD-games.txt" \
                   "$BASE/FreeBSD-pkg-bootstrap.txt" \
                   "$BASE/FreeBSD-rc.txt" \
                   "$BASE/FreeBSD-rescue.txt" \
                   "$BASE/FreeBSD-src-sys.txt" \
                   "$BASE/FreeBSD-src.txt"

# System

DEST_DIR="$PKGS_BY_TYPE/system"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-csh.txt"

## Boot

DEST_DIR="$PKGS_BY_TYPE/system/boot"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-bootloader.txt" \
                   "$BASE/FreeBSD-efi-tools.txt"

## Device management

DEST_DIR="$PKGS_BY_TYPE/system/device-management"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-devd.txt" \
                   "$BASE/FreeBSD-devmatch.txt"

## Kernel images

DEST_DIR="$PKGS_BY_TYPE/system/kernel"
mkdir -pv "$DEST_DIR"

KEY="kernel"
find "$BASE" -name "*$KEY*" -exec mv -vt "$DEST_DIR" {} +

## Logging

DEST_DIR="$PKGS_BY_TYPE/system/logging"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-newsyslog.txt" \
                   "$BASE/FreeBSD-syslogd.txt"

## Storage

DEST_DIR="$PKGS_BY_TYPE/system/storage"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-autofs.txt" \
                   "$BASE/FreeBSD-ccdconfig.txt" \
                   "$BASE/FreeBSD-geom.txt" \
                   "$BASE/FreeBSD-nvme-tools.txt" \
                   "$BASE/FreeBSD-ufs.txt" \
                   "$BASE/FreeBSD-zfs.txt"

# Utilities

DEST_DIR="$PKGS_BY_TYPE/util"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-acct.txt" \
                   "$BASE/FreeBSD-console-tools.txt" \
                   "$BASE/FreeBSD-dpv.txt" \
                   "$BASE/FreeBSD-quotacheck.txt"

## Editors

DEST_DIR="$PKGS_BY_TYPE/util/editor"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-ee.txt" \
                   "$BASE/FreeBSD-vi.txt"

## Power management

DEST_DIR="$PKGS_BY_TYPE/util/power"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-acpi.txt" \
                   "$BASE/FreeBSD-apm.txt" \
                   "$BASE/FreeBSD-smbutils.txt"

## Scheduling

DEST_DIR="$PKGS_BY_TYPE/util/scheduling"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-at.txt" \
                   "$BASE/FreeBSD-periodic.txt"

## Virtualization

DEST_DIR="$PKGS_BY_TYPE/virt"
mkdir -pv "$DEST_DIR"

mv -vt "$DEST_DIR" "$BASE/FreeBSD-bhyve.txt" \
                   "$BASE/FreeBSD-hyperv-tools.txt" \
                   "$BASE/FreeBSD-jail.txt"

rmdir -v "$BASE"

