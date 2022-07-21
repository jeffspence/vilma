# All paths will be relative to this file's location:
EXAMPLEDIR=$(dirname -- $0)


# Before beginning, let's clean up some intermediat files if they
# exits
if test -f ${EXAMPLEDIR}/ld_mat/example_schema.schema; then
	rm ${EXAMPLEDIR}/ld_mat/example_schema.schema
fi

# First we must build an LD matrix
# The --out-root option specifies that we want
# to store all of this in the ld_mat directory
# and all of the filenames will start with
# /example_schema
vilma make_ld_schema --logfile - \
	--out-root ${EXAMPLEDIR}/ld_mat/example_schema \
	--extract ${EXAMPLEDIR}/keep_variants.txt \
	--block-file ${EXAMPLEDIR}/blockfile.bed \
	--plink-file-list ${EXAMPLEDIR}/plink_file_list.txt \
	--ldthresh 0.8



# Now we estimate the optimal weights to use for polygenic scores
# This will load (the totally made up) results of a GWAS from
# example_data/example_gwas_sumstats.txt, and use
# the LD matrix we just computed.
# This will use ~81 mixture components in the prior.
# We will guess that this trait has a heritability of
# 0.2, and that the sample size for the GWAS was 300.000.
# In the output we will call the estimated effect sizes
# posterior_ukbb.
# We only want to use the variants listed in
# keep_variants.txt
vilma fit --logfile - \
	--sumstats ${EXAMPLEDIR}/example_data/example_gwas_sumstats.txt \
	--output ${EXAMPLEDIR}/example_vilma_run \
	--ld-schema ${EXAMPLEDIR}/ld_mat/example_schema.schema \
	--seed 42 \
	-K 81 \
	--init-hg 0.2 \
	--samplesizes 300e3 \
	--names ukbb \
	--learn-scaling \
	--extract ${EXAMPLEDIR}/keep_variants.txt

# example_vilma_run.estimates.tsv should now contain
# the posterior mean estimates you need to build
# a PGS using PLINK!

# example_vilma_run.estimates.tsv should also match
# copy_of_example_vilma_run.estimates.tsv if
# everything went well
