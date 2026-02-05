import warnings

import pandas as pd


def map_nw_ids(df, id_in, id_out, suffixes=('_a', '_b'),
               directed=False, agg=None):
    """Tables mapping between different protein/ORF/gene identifiers.

    All combinations of pairs of IDs are returned. Mappings are symmetric.

    Supported IDs are: orf_id, ensembl_gene_id, uniprot_ac
    Where orf_id refers to our internal ORF IDs.

    The ORF ID mappings come from the ORFeome annotation project where ORFs
    were sequence aligned to ensembl transcripts and proteins. The uniprot
    to ensembl mappings are from a file provided by UniProt.

    Note:
        The ensembl gene IDs and UniProt ACs are an unfiltered redundant set.
        You will probably want to filter the set after mapping.

    Args:
        df (DataFrame): table of protein-protein interactions,
                        one row for each pair
        id_in/id_out (str): gene/protein identifiers to map between
        suffixes (tuple(str, str)): of the two protein identifiers of each pair
        directed (bool): if True, retain the mapping of the two proteins,
                         if False, sort the IDs and drop duplicates
        agg (function): Optional. Custom aggregation function to choose a
                        single pair or combine multiple pairs when input pairs
                        are mapped to the same pair in the new ID. Function
                        must take a DataFrame and return a DataFrame with no
                        duplicate pairs.

    Returns:
        DataFrame: PPI dataset mapped to new protein/gene ID. All unique
                   combinations of the new ID are mapped to, so one pair in
                   the input dataset can map to multiple pairs in the output
                   and vice-versa.

    """
    id_in_a = id_in + suffixes[0]
    id_in_b = id_in + suffixes[1]
    id_out_a = id_out + suffixes[0]
    id_out_b = id_out + suffixes[1]
    id_map = load_id_map(id_in, id_out)
    out = df.copy()
    out = (pd.merge(out, id_map,
                    how='inner',
                    left_on=id_in_a,
                    right_on=id_in)
             .drop(columns=id_in)
             .rename(columns={id_out: id_out_a}))
    out = (pd.merge(out, id_map,
                    how='inner',
                    left_on=id_in_b,
                    right_on=id_in)
             .drop(columns=id_in)
             .rename(columns={id_out: id_out_b}))
    if out.loc[:, [id_out_a, id_out_b]].isnull().any().any():
        raise UserWarning('Unexpected missing values')
    if not directed:
        a = out[[id_out_a, id_out_b]].min(axis=1)
        b = out[[id_out_a, id_out_b]].max(axis=1)
        out[id_out_a] = a
        out[id_out_b] = b
    out = out.drop(columns=[id_in_a, id_in_b])
    pair_duplicates = out.duplicated(subset=[id_out_a, id_out_b], keep=False)
    if (out.duplicated(keep=False) != pair_duplicates).any() and agg is None:
        warnings.warn('Warning: mapping between gene/protein identifiers has '
                      'resulted in different pairs in the input ID being mapped to '
                      'the same pair in the output ID.\n'
                      'You may wish to use the `agg` argument to customize '
                      'the choice of which of the pair\'s infomation to keep or how '
                      'to combine the information from multiple pairs.')
    if agg is None:
        out = out.drop_duplicates(subset=[id_out_a, id_out_b])
    else:
        out = agg(out)
        if out.duplicated().any():
            raise ValueError('Problem with your agg function')
    cols = out.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    out = out.loc[:, cols]
    out = out.set_index(out[id_out_a].astype(str) +
                        '_' +
                        out[id_out_b].astype(str))
    return out


def _load_ensembl_to_uniprot():
    return pd.read_csv('../data/internal/ensembl_to_uniprot_mapping.tsv',
                       sep='\t')


def load_id_map(id_in, id_out):
    """Tables mapping between different protein/ORF/gene identifiers.

    All combinations of pairs of IDs are returned. Mappings are symmetric.

    Supported IDs are: orf_id, ensembl_gene_id, uniprot_ac, hgnc_id
    Where orf_id refers to our internal ORF IDs.

    The ORF ID mappings come from the ORFeome annotation project where ORFs
    were sequence aligned to ensembl transcripts and proteins. The uniprot
    to ensembl mappings are from a file provided by UniProt.

    Note:
        The ensembl gene IDs and UniProt ACs are an unfiltered redundant set.
        You will probably want to filter the set after mapping.

    TODO:
        - should I be checking for NULL values in the query????
        - implement uniprot_iso

    Args:
        id_in/id_out (str): gene/protein identifiers to map between

    Returns:
        DataFrame: two columns; id_in and id_out.

    """
    valid_ids = ['orf_id',
                 'ensembl_gene_id',
                 'uniprot_ac',
                 'hgnc_id',
                 'hgnc_symbol']
    for id_type in [id_in, id_out]:
        if id_type not in valid_ids:
            raise ValueError('Unsupported ID: ' + id_type +
                             '\nChoices are: ' + ', '.join(valid_ids))
    if id_in == id_out:
        raise ValueError('Invalid arguments: id_in == id_out')
    ids = set([id_in, id_out])
    if ids == {'uniprot_ac', 'ensembl_gene_id'}:
        return _load_ensembl_to_uniprot()
    else:
        raise NotImplementedError('just using uniprot <-> ensembl mapping for this project')
