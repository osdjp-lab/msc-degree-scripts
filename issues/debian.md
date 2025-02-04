# Debian

## Data extraction

### Multiple sources (DONE)

Problem:
Multiple sources of package metadata: 'Contents-amd64' and 'Packages'

Solution:
Split extracted data into directories based on origin

### Missing contents listings

Problem:
Missing contents listings for certain packages (0ad-data, 0ad-data-common, 2ping, 2to3, etc.)

Solution:
If needed missing data in https://ftp.debian.org/debian/dists/bookworm/main/Contents-all.gz

### Incomplete split of package listings by directory of origin

Problem:
Subdirectories not handled suitably in scripts

Solution:
If needed extract into seperate shell script and utilize the current awk script

### Some unnecessary fields from 'Packages' where not removed.

Problem:
Before initial split of 'Packages' some unnecessary fields where not removed.

Solution:
Find and remove unnecessary fields/lines.

### Incorrect creation of 'pkg-contents' (DONE)

Problem:
In 'Contents-amd64' some files are assigned multiple packages split by a comma also some file names contain spaces

Solution:
Process 'Contents-amd64' while taking the formatting issues into consideration.

### Trailing spaces in 'pkg-contents' (DONE)

Problem:
Trailing spaces in 'pkg-contents'.

Solution:
Strip trailing spaces from file before processing.

### Due to files with spaces incorrect generation of 'all-pkg.lst'

Problem:
File names with spaces in listing.

Solution:
Find work around.

