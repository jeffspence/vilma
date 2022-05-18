# In this script we will load the checkpoint saved
# at the end of the script in example.sh.
# If you have not run that script, start there.
# This example shows how to load a checkpoint
# and resume optimization from there.

# The key is the --load-checkpoint option. The first
# argument is the checkpoint .npz file and the second
# argument is the covariance .pkl file.

# vilma will always save a checkpoint of the final model
# when optimization is done.  Alternatively, one can save
# intermediate models by using the --checkpoint-freq option.
# Run vilma fit --help for more details.

vilma fit --logfile - \
	--sumstats example_data/example_gwas_sumstats.txt \
	--output checkpoint_example_vilma_run \
	--ld-schema ld_mat/example_schema.schema \
	--seed 42 \
	-K 81 \
	--init-hg 0.2 \
	--samplesizes 300e3 \
	--names ukbb \
	--learn-scaling \
	--extract keep_variants.txt \
	--load-checkpoint example_vilma_run.npz example_vilma_run.covariance.pkl
