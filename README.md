# Vilma

polygenic scores using variational inference on GWAS summary statistics from multiple cohorts


## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
    * [Building an LD Matrix](#building-an-ld-matrix)
    * [Building a Polygenic Score](#building-a-polygenic-score)
* [Output File Formats](#output-file-formats)
* [Example](#example)
* [Citation](#citation)


Installation
------------

`vilma` requires python v.3.7.11 or higher.
`vilma` makes use of a number of external packages
so it is recommended to install `vilma` in a virtual
environment.

If using
[conda](https://conda.io/en/master/)
this can be accomplished by running

```
conda create -n my-vilma-env
conda activate my-vilma-env
```

If using `virtualenv` run:

```
virtualenv my-vilma-env
source my-pyrho-env/bin/activate
```

Note that activating the virttual environment will
be required before running `vilma` (e.g., if you open
a new terminal, or deactivate `my-vilma-env`).

First, `vilma` makes use of 
[hdf](https://www.hdfgroup.org)
in order to (optionally) store large matrices on disk.  If you
do not have hdf installed, it can be installed using
your systems's package manager, such as
```apt-get```, ```yum```, ```conda```, ```brew``` etc...

For example, on Ubuntu run:

```
sudo apt-get install libhdf5-serial-dev
```

You should then be able to clone and install `vilma` by running:

```
git clone https://github.com/jeffspence/vilma.git vilma
pip install vilma/
```

Note that this will create a directory `vilma/` in your current
working directory.

If you have
[pytest](https://docs.pytest.org/)
installed, you can check that everything is running smoothly
by running

```python -m pytest vilma/tests/test.py```

The first time you run `vilma`, `numba` will compile a number of functions.
This should take about a minute, but will only happen the first time `vilma`
is run.

Usage
-----

`vilma` has a command line interface and consists of two separate commands.
`vilma` works by combining GWAS summary statistics from one or more cohorts
with LD matrices for each of those cohorts in order to estimate a distribution of
effect sizes as well as posterior effect size estimates for each variant.

The GWAS summary statistics must be
provided by the user. These can be obtained from publicly available summary
statistics, or by running association testing software (e.g.,
[PLINK v1.9](https://www.cog-genomics.org/plink)
or
[PLINK v.2.0](https://www.cog-genomics.org/plink/2.0/)
).

The other inputs to the model are the LD matrices in each cohort which can
be constructed using `vilma make_ld_schema` as described
[below](#building-an-ld-matrix).

Once we have our summary statistics, and our LD matrices, we are ready
to simultaneously fit a distribution of effect sizes and build a
polygenic score using `vilma fit`.  This is described in
[Building a Polygenic Score](#building-a-polygenic-score).

The outputs of this process are described in
[Output File Formats](#output-file-formats).


### Building an LD Matrix

N.B. that in version 0.0.2 the file format for matrices with pre-computed
SVDs changed.  This results in LD matrices using up about half as much
memory on disk, but unfortunately requires that LD matrices built using
version 0.0.1 will need to be recomputed if the `--ldthresh` option was used.

The LD matrix for each cohort is (in principle) a `num_snps` x `num_smnps` matrix
where entry `i, j` is the correlation between genotypes at SNP `i` and SNP `j`.
In practice, `vilma` will be run on millions of SNPs and performing computations
with such a large matrix is prohibitive. As such, we build a sparse block-diagonal
matrix by dividing the genome into different "LD blocks" and assuming that the
correlation between SNPs in different blocks is zero.

In practice, we pre-compute this matrix and store it as a series of `numpy`
matrices that correspond to the submatrices along the diagonal, and a corresponding 
series of variant files that list which SNPs are included in each submatrix.
An overall "manifest" file contains the paths of all of these matrix and variant
files.

`vilma` can construct such a block diagonal matrix from 
[PLINK v1.9](https://www.cog-genomics.org/plink/1.9/formats)
format genotype data (`.bim`, `.bed`,  and `.fam` files) and
a file containing the limits of separate LD blocks
(as a `.bed` file, but
[this bed](https://genome.ucsc.edu/FAQ/FAQformat.html#format1),
not the PLINK `.bed`). 

To build an LD matrix, run

```
vilma make_ld_schema --logfile <log_file_name> \
    --out-root <output_path_root> \
    --extract <snp_file> \
    --block-file <block_file> \
    --plink-file-list <plink_path_file> \
    --ldthresh <t>
```

`<log_file_name>` is the name of a file to store logging information that `vilma`
will output while running.  For `std-out` use `-`.

`<output_path_root>` will determine where the output is stored. The manifest file
which will be used
[below](#building-a-polygenic-score)
will be saved to `<output_path_root>.schema`.  All of the matrix and variant files
will be saved to `<output_path_root>_{chromosome}:{block_number}.npy` and
`<output_path_root>_{chromosome}:{block_number}.var`.

`<snp_file>` should contain a whitespace delimited file with a column labeled
"ID" that contains which SNPs should be read in and used when building the LD
matrix. This is optional, and if excluded, then all SNPs will be used
when building an LD matrix.

`<block_file>` is a `.bed` format file (whitespace delimited with columns
`chromosome` `start` `end`). The chromosome names should match those in
the genotype data. Each line corresponds to an LD block, with `start` being
the 0-indexed, inclusive start of the block in base pairs and `end` being
the 0-indexed, exclusive end of hte block in base pairs.  That a line
`chr1 100 1000` indicates that any SNPs on chromosome `chr1` at positions
(1-indexed, i.e., matching plink)
`101`, `102`, ..., `1000` will be treated as being in the same LD block.
These blocks must be non-overlapping.

`<plink_path_file>` is a file which contains the "basename" of PLINK1.9
format genotype data for a single chromosome on each line.  That is,
if the first line is `<path_to_plink_data_for_chromosome_1>` and the
second line is `<path_to_plink_data_for_chromosome_2>`, then `vilma`
will look for `<path_to_plink_data_for_chromosome_1>.bim`,
`<path_to_plink_data_for_chromosome_1>.bed`,
and `<path_to_plink_data_for_chromosome_1>.fam`, split the genotype
data into the blocks specified by all of the rows of `<block_file>`
that start with the chromosome identifier present in `<path_to_plink_data_for_chromosome_1>.bim`,
and then compute correlation matrices.  Then it will load the `.bim`, `.bed`, and `.fam` files
that start with `<path_to_plink_data_for_chromosome_2>` and so on.

`--ldthresh <t>` is optional.
[Later](#building-a-polygenic-score) we will perform singular value decompositions (SVDs) on
each submatrix of the LD matrix in order to denoise it. This can take some time, however,
so if multiple polygenic scores will be built using the same LD matrix, it makes sense
to run these SVDs up front and store the results. To do this, include the `--ldthresh <t>` option.
Setting a threshold of `<t>` guarantees that SNPs with an `r^2` between them of `<t>` or
smaller will be treated as linearly independent.  
Smaller values of `<t>` will result in lower memory usage and faster runtimes.
Larger values of `<t>` should result in more accurate polygenic scores up to a certain point --
if `<t>` is too close to one, noisy components of the estimated LD matrix will start to be included.
Setting `<t>` to `0.8` seems to perform well in practice.

For a detailed descriptions of all options, run

```
vilma make_ld_schema --help
```

### Building a Polygenic Score

Once we have an LD Matrix as computed
[above](#building-an-ld-matrix),
we are ready to fit the model to data.  This is done using `vilma fit`.  A standard usage is
```
vilma fit --ld-schema <comma_delimited_list_of_manifest_files> \
    --sumstats <comma_delimited_list_of_sumstatfiles> \
    --output <output_root> \
    -K <components> \
    --ldthresh <t> \
    --init-hg <comma_delimited_list_of_hgs> \
    --samplesizes <comma_delimited_list_of_Ns> \
    --names <comma_delimited_list_of_cohort_names> \
    --learn-scaling \
    --logfile <log_file> \
    --extract <snp_file>
```

We detail the different options below. 

`<comma_delimited_list_of_manifest_files>` is a comma separated list of paths to LD 
matrix manifests (one for each cohort) as computed in
[Building an LD Matrix](#building-an-ld-matrix). Specifically, if `vilma make_ld_schema`
was run with option `--out-root <output_path_root>` then `<output_path_root>.schema` should
be passed to `vilma fit`.

`<comma_delimited_list_of_sumstatfiles>` is a comma separated list of paths to
summary statistics files. These files are the summary of GWAS associations.
Each file must contain a column `ID` with the name of the SNP (to match to the
LD matrices computed above), a column labeled either `BETA` or `OR` that contains
the estimated marginal effect size (or odd ratio) for this SNP (`OR` should be for case-control
data), and a column labeled `SE` that contains the standard error of the GWAS
marginal effect size estimate (or log odds ratio for case-control data). 
To ensure that direction of effect (i.e., which SNP has a positive vs. negative effect) matches
the correlations of alleles in the LD matrix, we must determine which allele was used
in the assocation test. To that end, the summary stats file must contain a column labeled
`A1`. Then, to be compatible with either PLINK1.9 and PLINK2.0, there must either be
two columns labeled `REF` and `ALT` (PLINK 1.9) or a column labeled `A2` (PLINK 2.0).

`<output_root>` is the base name for all of the output generated by `vilma fit`.
There will be a number of outputs described
[below](#output-file-formats). `<output_root>.covariance.pkl` will contain the
covariance matrices of the mixture components used by `vilma`.
`<output_root>.npz` will contain the complete fit model.
`<output_root>.estimates.tsv` will contain the posterior mean effect sizes,
which are the optimal weights when building a polygenic score.

`<components>` determines how many mixture components will be used in the prior.
More components will result in a better fit, but a longer runtime.  The actual
number of components used is based on `<components>` but in general will be
larger. This is so that the space of potential variants will be well-covered
and that using the same `<components>` values for one or two cohorts will
cover the space of covariances comparably well.  As a result, for a particular
value of `<components>` there will be many more mixture components in a two
cohort model than in a one cohort model.

`--ldthresh <t>` sets how accurately to approximate the LD matrices.
`vilma` performs singular value decompositions (SVDs) on
each submatrix of the LD matrix in order to denoise it.
Setting a threshold of `<t>` guarantees that SNPs with an `r^2` between them of `<t>` or
smaller will be treated as linearly independent.  
Smaller values of `<t>` will result in lower memory usage and faster runtimes.
Larger values of `<t>` should result in more accurate polygenic scores up to a certain point --
if `<t>` is too close to one, noisy components of the estimated LD matrix will start to be included.
Setting `<t>` to `0.8` seems to perform well in practice.

`<comma_delimited_list_of_hgs>` is used in initializing `vilma fit`. This
should be a comma delimited list of the approximate heritabilities of
the trait in each cohort.  As this is only used for initialization it is
not crucial to get this exactly correct.

`<comma_delimited_list_of_Ns>` is used in initializing `vilma fit`. This
should be a comma delimited list of the (effective) sample sizes of
the GWAS in each cohort. As this is only used for initialization it
is not crucial for this to be exact.

`<comma_delimited_list_of_cohort_names>` is used in the output.
In particular, `<output_root>.estimates.tsv` will contain a column
for each cohort with the posterior mean estimate of the effect size for each SNP
in that cohort. For example if we use `--names ukbb,bbj` then
there will be columns `posterior_ukbb` and `posterior_bbj` in
`<output_root>.estimates.tsv`. The column `posterior_ukbb`  will contain estimates
from using the first LD matrix, the first summary stats file, the first init-hg,
and so on. The default is "0", "1", ...

`--learn-scaling` causes `vilma fit` to learn an analog of the LDSC intercept term
that accounts for improperly calibrated standard errors in the GWAS (e.g.,
over-correcting or under-correcting for population structure).

`<log_file_name>` is the name of a file to store logging information that `vilma`
will output while running.  For `std-out` use `-`.


`<snp_file>` should contain a whitespace delimited file with a column labeled
"ID" that contains which SNPs should be read in and used when building the 
polygenic score.  
To ensure that direction of effect (i.e., which SNP has a positive vs. negative effect) matches
the correlations of alleles in the LD matrix, we must determine which allele was used
in the assocation test. To that end, the summary stats file must contain a column labeled
`A1`. Then, to be compatible with either PLINK1.9 and PLINK2.0, there must either be
two columns labeled `REF` and `ALT` (PLINK 1.9) or a column labeled `A2` (PLINK 2.0).

Polygenic scores can then be computed for genotype data using the weights
inferred by `vilma` by using 
[Allelic scoring in PLINK v1.9](https://www.cog-genomics.org/plink/1.9/score)
or
[Linear scoring in PLINK v2.0](https://www.cog-genomics.org/plink/2.0/score).

For a detailed description of these options (and additional options), run

```
vilma fit --help
```

Output File Formats
------------

`vilma fit` produces three types of output files.

`<out_base>.estimates.tsv` contains the posterior mean effect sizes estimates for each cohort.
This is what should be passed to PLINK for scoring individuals (i.e., computing individuals'
polygenic scores).  This file is tab-delimited and will contain 3 columns plus
one column per cohort. The column `ID` contains the SNP IDs.  The column `A1` contains the
allele which corresponds to the effect (i.e., the effect is the effect of each additional
copy of allele `A1`) and `A2` contains the other allele.  The columns `posterior_<cohort_name>`
contain the posterior mean estimate of the effect of each copy of the `A1` allele (in liability
for dichotomous traits).

`<out_base>.covariance.pkl` is a `python` `pickle` file that contains the covariance matrices
(called ∑ in the paper) that comprise the component distributions of the prior.
In python these can be accessed using

```
import pickle
matrices = pickle.load(open('<out_base>.covariance.pkl', 'rb'))
matrices[0][0]  # the first covariance matrix
len(matrices[0])  # the total number of covariance matrices
```

`<out_base>.npz` is a `numpy` `npz` file that contains the fit model. There
are three arrays in this file. `vi_mu` is a `[num_components][num_cohorts][num_snps]`
dimensional array that contains the variational distribution means.  That is, 
`vi_mu[k][p][i]` is the posterior mean value of SNP `i` in cohort `p` given
that we are looking at component `k`. `vi_delta` is a `[num_snps][num_components]`
dimensional array that containsthe mixture weights of the different mixture
components for each SNP.  That is, the probability under the posterior that
the effect size for SNP `i` came from component `k` is `vi_delta[i][k]`. Furthermore,
this means that the overall posterior mean effect for a SNP in population `p` is
`vi_delta[i] @ vi_mu[:, p, i]`.  Finally, `hyper_delta` is a
`[num_annotations][num_components]` dimension array, with the (learned) prior mixture
weights for the different components of the prior.  That is `hyper_delta[a][k]` is the
prior probability that a SNP with annotation `a` comes from mixture component `k`. For
all of these arrays, the order of the component distributions matches that in
`<out_base>.covariance.pkl`, the order of the SNPs matches the file passed as the
`--extract` argument to `vilma fit`, and the order of the cohorts is the order in which
the summary statistics files, LD matrices, etc... were passed to `vilma fit`.



Example
-------

For an example workflow running `vilma` see `example.sh`
in the `example/` directory, where an LD matrix is built
from genotype data using `vilma make_ld_schema` and then the model is fit using
`vilma fit`.


Citation
--------

If you use `vilma` please cite

[Spence, J. P., Sinnott-Armstrong, N., Assimes, T. L., and Pritchard, J. K.
A flexible modeling and inference framework for estimating variant effect sizes
from GWAS summary statistics. _bioRxiv_](https://www.biorxiv.org/content/10.1101/2022.04.18.488696v1)

