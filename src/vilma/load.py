"""
Utilities for loading and manipulating GWAS summary stats and LD.

functions:
    load_variant_list: Read in a list of variants to use for analysis
    load_annotations: Read an annotation file and merge with variants
    load_sumstats: Load GWAS summary stats and merge with variants
    load_ld_from_schema: Load a block LD matrix using a schema file
        and match to variants
"""
import logging
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
import h5py
from vilma.matrix_structures import LowRankMatrix
from vilma.matrix_structures import BlockDiagonalMatrix


def load_variant_list(variant_filename):
    """Read in a list of variants from `variant_filename`"""
    variants = pd.read_csv(variant_filename,
                           header=0,
                           delim_whitespace=True).drop_duplicates()
    if 'ID' not in variants.columns:
        raise ValueError('Variant file must contain a column labeled ID')
    if 'A1' not in variants.columns:
        raise ValueError('Variant file must contain a column labeled A1')
    if 'A2' not in variants.columns:
        if 'REF' not in variants.columns or 'ALT' not in variants.columns:
            raise ValueError('Variant file must contain a column labeled A2')
        variants['A2'] = variants['REF'].copy()
        flip = variants['A1'] == variants['REF']
        variants.loc[flip, 'A2'] = variants.loc[flip, 'ALT'].copy()

    variants = variants[['ID', 'A1', 'A2']]

    return variants


def load_annotations(annotations_filename, variants):
    """Read `annotations_filename`  and match annotations to `variants`"""
    if annotations_filename is None:
        return np.ones((variants.shape[0], 1)), []

    dframe = pd.read_csv(annotations_filename,
                         header=0,
                         delim_whitespace=True)

    if 'ID' not in dframe.columns:
        raise ValueError('Annotation file must contain a column labeled ID')
    if 'ANNOTATION' not in dframe.columns:
        raise ValueError('Annotation file must contain a column labeled '
                         'ANNOTATION')

    dframe = pd.merge(variants, dframe, on='ID', how='left')
    dframe = pd.DataFrame(dframe['ANNOTATION'])
    logging.info('%d out of %d total variants are missing annotations',
                 dframe['ANNOTATION'].isna().sum(),
                 dframe.shape[0])

    denylist = np.where(dframe['ANNOTATION'].isna())[0].tolist()
    dframe.loc[dframe['ANNOTATION'].isna(), 'ANNOTATION'] = 0
    return pd.get_dummies(dframe['ANNOTATION'],
                          dummy_na=False).to_numpy(), denylist


def load_sumstats(sumstats_filename, variants):
    """Load summary stats from `sumstats_filename` and match to `variants`"""
    sumstats = pd.read_csv(sumstats_filename,
                           header=0,
                           delim_whitespace=True)

    if 'ID' not in sumstats.columns:
        raise ValueError('Summary Statistics File must contain a column '
                         'labeled ID')

    if 'A1' not in sumstats.columns:
        raise ValueError('Summary Statistics File must contain a column '
                         'labeled A1')

    if 'A2' not in sumstats.columns:
        if 'REF' not in sumstats.columns or 'ALT' not in sumstats.columns:
            raise ValueError('If summary statistics file does not contain '
                             'a column labeled A2, then it must contain REF '
                             'and ALT columns.')
        sumstats['A2'] = sumstats['REF'].copy()
        flip = sumstats['A1'] == sumstats['REF']
        sumstats.loc[flip, 'A2'] = sumstats.loc[flip, 'ALT'].copy()

    if 'SE' not in sumstats.columns:
        raise ValueError('Summary Statistics File must contain a column '
                         'labeled SE')

    if 'BETA' not in sumstats.columns:
        if 'OR' not in sumstats.columns:
            raise ValueError('Summary stat file needs to contain either'
                             'BETA or OR filed.')
        sumstats['BETA'] = np.log(sumstats.OR)

    sumstats = pd.merge(variants, sumstats, on='ID', how='left')
    missing = np.logical_or(np.isnan(sumstats.BETA.values),
                            np.isnan(sumstats.SE.values))
    stay_allele = ((sumstats.A1_x == sumstats.A1_y)
                   & (sumstats.A2_x == sumstats.A2_y))
    flip_allele = ((sumstats.A1_x == sumstats.A2_y)
                   & (sumstats.A1_y == sumstats.A2_x))
    missing = (sumstats.BETA.isna()
               | sumstats.SE.isna()
               | ((~stay_allele) & (~flip_allele)))
    logging.info('%d out of %d total variants are missing sumstats',
                 missing.sum(),
                 sumstats.shape[0])
    logging.info('%d alleles have been flipped',
                 (flip_allele).sum())
    sumstats.loc[missing, 'BETA'] = 0.
    sumstats.loc[missing, 'SE'] = 1.
    sumstats.loc[flip_allele, 'BETA'] = -sumstats.loc[flip_allele, 'BETA']
    return sumstats, np.where(missing)[0].tolist()


def load_ld_from_schema(schema_path, variants, denylist, ldthresh, mmap=False):
    """
    Load block matrix using specified schema, and filter to variants.

    args:
        schema_path: path to file containing the schema manifest
        variants: the set of variants at which we want LD
        denylist: Variants which should be treated as missing for the purposes
            of this LD matrix.
        ldthresh: LD threshold for low rank approximations to the LD matrix. A
            threshold of t guarantees that SNPs with r^2 lower than t
            will be linearly independent.
        mmap: Boolean to indicate whether to store LD matrix on disk (may be
            slow).  Defaults to storing LD matrix in memory (faster but more
            memory intensive).

    returns:
        A vilma.matrix_structures.BlockDiagonalMatrix containing the LD matrix
        ordered in the order of `variants`.

    """

    svds = []
    perm = []
    var_reidx = variants.set_index('ID')
    var_reidx['old_idx'] = np.arange(var_reidx.shape[0])
    total_flipped = 0
    hdf_file = None
    if mmap:
        hdf_file = h5py.File(TemporaryFile(), 'w')
    with open(schema_path, 'r') as schema:
        for line in schema:
            snp_path, ld_path = line.split()
            # relative path:
            if snp_path[0] != '/':
                snp_path = ('/'.join(schema_path.split('/')[:-1])
                            + '/' + snp_path)
                ld_path = ('/'.join(schema_path.split('/')[:-1])
                           + '/' + ld_path)

            snp_metadata = pd.read_csv(snp_path,
                                       header=None,
                                       delim_whitespace=True,
                                       names=['ID', 'CHROM', 'BP',
                                              'CM', 'A1', 'A2'])

            ld_shape = (snp_metadata.shape[0], snp_metadata.shape[0])

            logging.info('LD matrix shape: %s', (ld_shape,))

            variant_indices = np.copy(
                snp_metadata.ID.isin(variants.ID).to_numpy()
            )
            if np.sum(variant_indices) > 0:
                kept_ids = snp_metadata.ID[variant_indices]
                idx = var_reidx.loc[kept_ids].old_idx.to_numpy().flatten()
                keep = np.isin(idx, denylist, invert=True)
                to_change = np.where(variant_indices)[0][~keep]
                variant_indices[to_change] = False
                kept_ids = kept_ids.iloc[keep]
                idx = idx[keep]
                if len(idx) == 0:
                    continue
                signs = np.ones(len(idx))
                stay = [
                    (xa1 == ya1) and (xa2 == ya2)
                    for xa1, ya1, xa2, ya2 in
                    zip(variants['A1'].iloc[idx].to_numpy(),
                        snp_metadata['A1'].iloc[variant_indices].to_numpy(),
                        variants['A2'].iloc[idx].to_numpy(),
                        snp_metadata['A2'].iloc[variant_indices].to_numpy())
                ]
                stay = np.array(stay)
                flip = [
                    (xa1 == ya1) and (xa2 == ya2)
                    for xa1, ya2, xa2, ya1 in
                    zip(variants['A1'].iloc[idx].to_numpy(),
                        snp_metadata['A1'].iloc[variant_indices].to_numpy(),
                        variants['A2'].iloc[idx].to_numpy(),
                        snp_metadata['A2'].iloc[variant_indices].to_numpy())
                ]
                flip = np.array(flip)
                total_flipped += flip.sum()
                mismatch = np.logical_and(~flip, ~stay)
                if len(idx[~mismatch]) == 0:
                    continue
                signs[flip] = -1
                ld_matrix = np.copy(np.load(ld_path))
                if len(ld_matrix.shape) == 0:
                    ld_matrix = ld_matrix[None, None]
                if ld_matrix.shape[0] == ld_matrix.shape[1]:
                    logging.info('Proportion of variant indices '
                                 'being used: %e',
                                 np.mean(variant_indices))

                    accepted_matrix = np.copy(
                        ld_matrix[np.ix_(variant_indices, variant_indices)]
                    )
                    accepted_matrix = accepted_matrix * np.outer(signs, signs)
                    accepted_matrix = accepted_matrix[np.ix_(~mismatch,
                                                             ~mismatch)]
                else:
                    if ld_matrix.shape[0] < ld_matrix.shape[1]:
                        raise ValueError('Bad LD matrix.')
                    num_snps = (ld_matrix.shape[0] - 1)
                    if num_snps != variant_indices.shape[0]:
                        raise ValueError('Bad LD matrix.')
                    u_mat = np.copy(ld_matrix[0:num_snps])
                    s_vec = np.copy(ld_matrix[num_snps])

                    u_mat = u_mat[variant_indices, :]
                    u_mat = signs.reshape((-1, 1)) * u_mat
                    u_mat = np.copy(u_mat[~mismatch])
                    accepted_matrix = (u_mat * s_vec).dot(u_mat.T)

                perm.append(idx[~mismatch])
                svds.append(
                    LowRankMatrix(accepted_matrix,
                                  ldthresh,
                                  hdf_file=hdf_file)
                )

    # Need to add in variants that are not in the LD matrix
    # Set them to have massive variance
    if len(perm) > 0:
        perm = np.concatenate(perm)
    else:
        perm = np.array([], dtype=float)
    missing = set(np.arange(variants.shape[0]).tolist()) - set(perm.tolist())
    missing = np.array(list(missing), dtype=int)
    logging.info('Loaded a total of %d variants.', variants.shape[0])
    logging.info('Missing LD info for %d variants. They will be ignored '
                 'during optimization.', len(missing))
    logging.info('The alleles did not match for %d variants. They were '
                 'flipped', total_flipped)
    perm = np.concatenate([perm, missing])
    if not np.all(perm == np.arange(len(perm))):
        logging.info('The variants in the extract file and the variants in '
                     'the LD matrix were not in the same order.')
    perm = np.array(perm)
    block_mat = BlockDiagonalMatrix(svds, perm=perm, missing=missing)

    return block_mat
