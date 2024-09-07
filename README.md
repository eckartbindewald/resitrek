# Protein Function Prediction with protrekfun


## Installation

### Change current working directory to a location for installing ProTrek

```
cd protrekfun/..
source protrekfun/install_protrek.sh
```
This will install the ProTrek repository under directory name ProTrek. The script prints the absolute pathname of that installation directory,
that should be used in subsequent steps

## How to run

This program uses the ProTrek model to determine protein function one a residue level.
Usage:

`python resitrek.py <PROTREK_PATH> <UNIPROT-ID> <3D-STRUCTURE>`

With 

* `PROTREK_PATH`: Path to installation directory of the ProTrek repository <https://github.com/westlake-repl/ProTrek>
* `UniProt-ID`: The Uniprot ID number for this protein
* `3D-STRUCTURE`: Path name to a protein 3D structure in mmcif format

    
### Example:
    

Let's assume that the installation directories of `resitrek` and `ProTrek` are part of the same directory.
In other words, from within directory `resitrek` the relative pathname to the ProTrek installation directory
is `../ProTrek`.

Then, the program can be called with:
```
 cd resitrek
 python scripts/resitrek.py ../ProtTrek P23946 example/example/8ac8.cif 
    
```

## Interpretation

The program outputs line by line a P-value for the chance that the currently observed sequence region is optimal with respect to randomly chosen alternatives.
In other words, a low number (like < 0.01) is indicative that the current position is important for the observed function.

## Who to talk to

* Eckart Bindewald <bindewald@hood.edu>
