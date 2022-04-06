from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import logging
import sys
from tempfile import TemporaryFile
from matrix_structures import LowRankMatrix
from matrix_structures import BlockDiagonalMatrix
import h5py


logger = logging.getLogger()

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


def load_variant_list(variant_path):
    """Read in a list of variants to use for analysis."""
    return pd.read_table(variant_path).drop_duplicates()


def load_annotations(annotations_file, variants):
    """Read annotation file. Just make 1s for now."""
    if annotations_file is None:
        return np.ones((variants.shape[0], 1))
    df = pd.read_table(annotations_file)
    df = pd.merge(variants, df, on='ID', how='left')
    df = pd.DataFrame(df['ANNOTATION'])
    logging.info('%d out of %d total variants are missing annotations',
                 df['ANNOTATION'].isna().sum(),
                 df.shape[0])
    df.loc[df['ANNOTATION'].isna(), 'ANNOTATION'] = np.random.choice(
        np.nanmax(df['ANNOTATION'].values)+1,
        size=df['ANNOTATION'].isna().sum()
    )
    return pd.get_dummies(df['ANNOTATION'],
                          dummy_na=False).to_numpy()


def load_sumstats(sumstats_path, variants):
    sumstats = pd.read_table(sumstats_path)
    sumstats = pd.merge(variants, sumstats, on='ID', how='left')
    if 'BETA' not in sumstats.columns:
        if 'OR' not in sumstats.columns:
            raise IOError('Summary stat file needs to contain either'
                          'BETA or OR filed.')
        sumstats.BETA = np.log(sumstats.OR)
    missing = np.logical_or(np.isnan(sumstats.BETA.values),
                            np.isnan(sumstats.SE.values))
    stay_allele = sumstats.A1_x == sumstats.A1_y
    logging.info('%d out of %d total variants are missing sumstats',
                 missing.sum(),
                 sumstats.BETA.values.shape[0])
    logging.info('%d alleles do not match.  Those have been flipped',
                 (~stay_allele).sum() - missing.sum())
    sumstats.loc[missing, 'BETA'] = 0.
    sumstats.loc[missing, 'SE'] = 1.
    sumstats.loc[~stay_allele, 'BETA'] = -sumstats.loc[~stay_allele, 'BETA']
    return sumstats, np.where(missing)[0]


def load_ld_from_schema(schema_path, variants, denylist, t, mmap):
    """Load block matrix using specified schema, and filter to variants.

    Schema format:
        SNPMetadataPath<tab>LDPanelMatrix
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

            snp_metadata = pd.read_table(snp_path,
                                         header=None,
                                         names=['ID', 'CHROM', 'BP',
                                                'CM', 'A1', 'A2'])

            ld_shape = (snp_metadata.shape[0], snp_metadata.shape[0])

            logger.info('LD matrix shape: %s' % (ld_shape,))

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
                perm.append(idx)
                signs = np.ones(len(idx))
                stay = [
                    v1 == v2 for v1, v2 in
                    zip(variants['A1'].iloc[idx].to_numpy(),
                        snp_metadata['A1'].iloc[variant_indices].to_numpy())
                ]
                stay = np.array(stay)
                assert np.all(
                    variants['A2'].iloc[idx].iloc[stay].to_numpy()
                    ==
                    snp_metadata['A2'].iloc[
                        variant_indices].iloc[stay].to_numpy()
                )
                flip = ~stay
                total_flipped += flip.sum()
                assert np.all(
                    variants['A2'].iloc[idx].iloc[flip].to_numpy()
                    ==
                    snp_metadata['A1'].iloc[
                        variant_indices].iloc[flip].to_numpy()
                )
                assert np.all(
                    variants['A1'].iloc[idx].iloc[flip].to_numpy()
                    ==
                    snp_metadata['A2'].iloc[
                        variant_indices].iloc[flip].to_numpy()
                )
                signs[flip] = -1
                signs = np.outer(signs, signs)
                ld_matrix = np.copy(np.load(ld_path))
                assert len(ld_matrix.shape) == 2 or len(ld_matrix.shape) == 0
                if len(ld_matrix.shape) == 0:
                    ld_matrix = ld_matrix[None, None]
                    assert np.allclose(ld_matrix, 1)
                assert ld_matrix.shape == ld_shape, (ld_matrix.shape, ld_shape)
                logger.info('Proportion of variant indices being used: %e'
                            % np.mean(variant_indices))

                accepted_matrix = np.copy(ld_matrix[np.ix_(variant_indices,
                                                           variant_indices)])
                accepted_matrix = accepted_matrix * signs
                assert accepted_matrix.shape == (np.sum(variant_indices),
                                                 np.sum(variant_indices))

                # if accepted_matrix.shape[0] > 42:
                #     u, s, v = svds(accepted_matrix, 42)
                # else:
                #     u, s, v = np.linalg.svd(accepted_matrix)

                # assert len(u.shape) == 2
                # assert len(s.shape) == 1
                # assert len(v.shape) == 2
                # assert u.shape[0] == np.sum(variant_indices), (
                #     schema_path, np.sum(variant_indices),
                #     u.shape, s.shape, v.shape
                # )
                # raw_mats.append(ld_matrix)
                # svds.append(matrix.SVD(accepted_matrix, 42))
                svds.append(
                    LowRankMatrix(accepted_matrix, t, hdf_file=hdf_file)
                )
                # us.append(u)
                # ss.append(s)
                # vs.append(v)
                # ld_mats.append(accepted_matrix)

    # Need to add in variants that are not in the LD matrix
    # Set them to have massive variance
    if len(perm) > 0:
        perm = np.concatenate(perm)
    else:
        perm = np.array([], dtype=float)
    missing = set(np.arange(variants.shape[0]).tolist()) - set(perm.tolist())
    missing = np.array(list(missing), dtype=int)
    logger.info('Loaded a total of %d variants.')
    logger.info('Missing LD info for %d variants. They will be ignored '
                'during optimization.', len(missing))
    logger.info('The alleles did not match for %d variants. They were '
                'flipped', total_flipped)
    perm = np.concatenate([perm, missing])

    if not np.all(perm == np.arange(len(perm))):
        logger.info('The variants in the extract file and the variants in the '
                    'LD matrix were not in the same order.')
    assert len(perm) == variants.shape[0]
    assert len(perm) == len(set(perm))
    perm = np.array(perm)
    bm = BlockDiagonalMatrix(svds, perm=perm, missing=missing)
    # bm = matrix.BlockMatrix([matrix.SVD(u, s, v)
    #                          for u, s, v in zip(*(us, ss, vs))],
    #                         perm=perm)
    # bm.raw_mats = raw_mats

    return bm
