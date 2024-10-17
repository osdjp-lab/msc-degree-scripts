# Detailed FreeBSD packages described

## Bundles

- runtime       - Base system
- utilities     - Non-vital programs and libraries

## Data

- caroot    - Transport Layer Security (TLS) certificates
- ipfilter  - ipf periodic configuration files
- locales   - Internationalization/localization (i18n/L10n) package
- mtree     - Mtree data
- syscons   - System console fonts, keymaps and screen output map files
- vt-data   - Virtual teminal fonts and keymaps
- zoneinfo  - Time zone database (tzdata)

## Development tools

- clang             - LLVM + Clang
- ctf-tools         - CTF (Compact C type Format) debug data tools
- dtrace            - Dynamic tracing framework
- dwatch            - Dtrace probe watcher
- elftoolchain      - ELF (Executable & Linkable Format) tools
- lldb              - LLVM debugger
- lld               - LLVM linker
- tests             - Kyua testing framework for infrastructure software + ATF (Automated Testing Framework)

## Libraries

- clibs                 - Core C libraries
- kerberos-lib          - Kerberos libraries
- lib9p                 - Plan9 libraries
- libarchive            - Library for reading and writing streaming archives
- libbegemot            - Library to hide the poll/select/setitimer interface from the user (rpoll functions)
- libblocksruntime      - LLVM blocks runtime library
- libbsdstat            - Library for managing and displaying periodically collected statistics
- libbsm                - OpenBSM library for event auditing
- libbz2                - Bzip2/libzip2 library for data compression
- libcasper             - Library for the control of application capabilities through the casper process
- libcuse               - Userland character device library
- libdwarf              - DWARF debugging data access library
- libevent1             - Asynchronous event notification library
- libexecinfo           - Backtrace program self-debugging library
- libipt                - Intel Processor Trace decoder library
- libldns               - DNS and DNSSEC library
- liblzma               - xz (formely LZMA) compression library
- libmagic              - File type guesser library (basis for file command)
- libpathconv           - Path conversion library
- librpcsec_gss         - Remote Procedure Call (RPC) security library that uses Generic Security Services (GSS)
- librss                - Receive-Side Scaling (RSS) library
- libsdp                - Bluetooth Service Discovery Protocol (SDP) user library
- libsqlite3            - SQLite library
- libstdbuf             - Standard I/O buffering modification library
- libstdthreads         - Lightweight threading interface library
- libthread_db          - Threads debugging library
- libucl                - Universal Configuration Library (UCL) for the generation and parsing of UCL objects
- libvgl                - Video Graphics Library (VGL) which uses syscons console driver
- libvmmapi             - Virtual Machine Memory (VMM) mapping library
- openssl-lib           - OpenSSL library

## Networking

### dns

- resolvconf    - Framework for managing multiple DNS configurations
- unbound       - A validating, recursive, caching DNS resolver

### driver

- cxgbe-tools   - Command line tools for the Chelsio Ethernet adapter driver
- mlx-tools     - Command line tools for managing Mellanox network adapters

### bluetooth   - Bluetooth control utilites
### bsnmp       - SNMP daemon
### dhclient    - DHCP client
### fetch       - Uniform Resource Locator (URL) file transfer tool and library
### ftpd        - File Transfer Protocol (FTP) daemon (deprecated)
### ftp         - File Transfer Protocol (FTP) client
### inetd       - Internet service dispatcher ("Super Server") daemon
### kerberos    - Kerberos utilities
### natd        - Network Address Translation (NAT) daemon
### netmap      - Utility for managing Virtual Local Ethernet (VALE) switches
### nuageinit   - Limited cloud init configuration
### ppp         - Point-to-point protocol (PPP) user implementation
### rcmds       - Local machine network status utilities (rwho, rwhod and ruptime)
### rdma        - Remote Direct Memory Access (RDMA) utilites
### ssh         - Secure Shell (SSH) utilities
### tcpd        - TCP/IP daemon wrapper utilities for monitoring and filtering requests
### telnet      - Telnet client
### yp          - NIS formerly Yellow Pages (YP) domain database utilites

### mail

- dma           - DragonFly Mail Agent - Mail Transport Agent (MTA)
- sendmail      - Sendmail - Mail Transport Agent (MTA)

### packet-filter

- blocklist     - Blacklistd daemon and library to block and release ports on demand
- ipf           - Packet filtering and Network Address Translation (NAT) libraries and utilites
- ipfw          - Firewall, traffic shaper, packet scheduling and in-kernel NAT management utility
- pf            - Packet Filter (PF) device management and logging utilites

### ssl

- certctl       - Tool for managing trusted and untrusted TLS certificates
- openssl       - OpenSSL Transport Layer Security (TLS) toolkit

### storage

- ggate         - GEOM framework gate protocol daemon and control utilities
- hast          - Highly Available Storage (HAST) protocol daemon and control utility
- iscsi         - Internet Small Computer Systems Interface (iSCSI) protocol daemon and control utilities
- nfs           - Network File System (NFS) protocol
- smbutils      - Server Message Block (SMB) utility

### wifi

- hostapd   - IEEE 802.11 network authentication utilities
- wpa       - Wi-Fi Protected Access (WPA) supplicant utilities

## Other (BSD specific packages)

- bsdinstall    - FreeBSD system install and sysadmins disk editor (sade)
- fwget         - Firmware installer
- games         - Games
- pkg-bootstrap - Bootstrap package for the installation of full featured pkg
- rc            - Rcorder shell script dependency ordering utility
- rescue        - Statically linked set of rescue tools
- src-sys       - FreeBSD kernel source code
- src           - FreeBSD userland source code

## System

### boot

- bootloader    - Bootloader
- efi-tools     - Extensible Firmware Interface (EFI) tools

### device-management

- devd          - Device state change daemon
- devmatch      - Utility for printing information about unattached devices

### csh         - Tcsh c-shell
### kernel

- kernel-generic-mmccam     - FreeBSD kernel with the MultiMediaCard (MMC) stack using the Common Access Method (CAM) framework enable (MMCCAM)
- kernel-generic            - FreeBSD kernel generic config
- kernel-minimal            - FreeBSD kernel minimal config

### logging

- newsyslog     - Utility for maintaining system log files at manageable sizes
- syslogd       - System logging daemon

### storage

- autofs        - Automated file system mounting daemon and utilities
- ccdconfig     - Concatenated Disk Driver (CCD) configuration utility
- geom          - GEOM framework utilities
- nvme-tools    - NVME Express control utility
- ufs           - Unix File System (UFS) utilities
- zfs           - Zettabyte File System (ZFS) utilities

## Utilities

### Editors

- ee    - Easy Editor (EE)
- vi    - Vi editor

### acct                - System Accounting utilities
### console-tools       - Keyboard system console (syscons) and virtual terminal (vt) control utilities
### dpv                 - Dialog Progress Viewer (DPV) tool and libraries
### quotacheck          - File system quota consistency checker
### power

- acpi          - Advanced Configuration and Power Interface (ACPI) control utilities
- apm           - Advanced Power Management (APM) utility

### Scheduling

- at            - One time scheduling utilities
- periodic      - Repetitive scheduling utility

## Virtualization
- bhyve             - Virtual machine hypervisor and control utilities
- hyperv-tools      - Windows Hyper-V integration support system utilities
- jail              - Jail (OS-level virtualisation) management utilities

