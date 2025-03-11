# Comparison notes

## Essential functions

FreeBSD and Debian:
pkg names -> pkg contents
pkg contents -> pkg names

## Comparability

Subdivide packages into ones where comparing the name or contents makes sense and those where it doesn't.

Doesn't make sense:
- Components with a similar or identical function but which are in no way the same (kernel, bootloader, virtualization)
- BSD specific utilities

Makes sense:
- Utilities
- Libraries
- Runtime
- Ports

## File placement

/bin vs /sbin vs /usr/bin vs /usr/sbin etc.

## Methods

### V1

Two files:
- one containing two columns the first with the names of FreeBSD packages the second with the names of the Debian packages which are their equivalents
- one containing two columns the first with the names of Debian packages the second with the names of the FreeBSD packages which are their equivalents

Issues:
- Doesn't work since the packages in the different systems do not for the most part allow for a simple 1:1 mapping

### V2

Comparison by name:
- Files: common, freebsd-only and debian-only pkg name listings
- Subdirectories for each matched package name with matched and freebsd-unmatched and debian-unmatched file lists

Comparison by contents:
- Directories for each FreeBSD base package
- Each directory containing: common, freebsd-only and debian-only file listings for the given package (comparison of FreeBSD package listing and debian all package listing)
- Extracted list of matched debian package names with count of matched files per matched debian package
- Subdirectory for each debian package with matched, freebsd-unmatched and debian-unmatched file lists

Issues:
- Comparisons yield an unwieldy amount of data
- Packages in different systems often have different names
- Contents listing may differ due to different full paths of files or different implementations of packages with the same functionality

