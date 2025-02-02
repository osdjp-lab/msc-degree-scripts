# Todo

## General

1. Consistency in for/while loop do placement and if then placement
2. If DEST_DIR does not exist make it
3. Conditional tool usage (bsdtar, sponge, etc.)
4. Verification message (print only basename)
5. Make regeneration of output easy (no duplication, errors, etc.)
6. Consistency of input checks and argument order (input then output)
7. In any command always put options first positional arguments last
8. Split long pipelines with pipe on newline
9. Performance stats.sh script for testing both individual debian and freebsd extraction scripts
10. Conditional execution of scripts (reduced runtime)
11. Replace ex with sed and sponge
12. Group output of extraction and comparison commands into subdirectories for ease of traversal and interpretation
13. Group numbered scripts from debian and freebsd by function or scope into subdirectories

## Debian

1. Debian add sort to generated file listing for consistancy
2. Extract packages with tag: "Essential: yes"
3. Add flag for package listing splitting (due to long execution time)

## FreeBSD

1. Gather meaningful descriptions of every FreeBSD base package in a single location (maybe by extracting man pages from each package into corresponding directories)
2. Deduplicate extracted man pages (by hash or cmp)
3. Fix package groupings both by type and by relevance
4. In freebsd/base/14-generate-pkg-contents-lst.sh substitute argument NAME_DIR for PKG_CONTENTS_LISTING
5. Scope split for base (full and simplified/deduplicated)

## Cross-reference

1. Deduplicate FreeBSD base packages (-dbg, -man, -dev, -lib32)
2. Group by type
3. Match by package name
4. Match by package contents (take single FreeBSD package listing find matches in Debian all file listing and place matched Debian packages in a file)
5. Match by package description (requires manual filtering)
6. Group by relevance (exclude irrelevant non security related packages)

For each consecutive step exclude previously found.

## Package content cross-reference

1. Use comm to split the FreeBSD and Debian package content listings into 3 subfiles: common, only-freebsd and only-debian
2. Recover package information about matched listed files from original debian listing (add back the missing package name column)
3. Extract all matched package names on the basis of matched file listings
4. Handle missing file on the basis of known found packages
5. If not found manual search

## Comparison

1. By name (priority, category, tags)
2. By contents

