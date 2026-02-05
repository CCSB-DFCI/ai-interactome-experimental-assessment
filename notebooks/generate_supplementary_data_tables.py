# %%
import os
from pathlib import Path
import json
import configparser

import numpy as np
import pandas as pd
from Bio import SeqIO
from goatools.obo_parser import GODag
import getpass
import keyring
import pymysql

from utils import ccsblib_config_dir

from datasets import (load_non_dubious_orfs, 
                      load_all_orfs,
                      load_pubmed_to_date_mapping,
                      load_i3d_exp_17, 
                      load_I3D_exp_24,
                      load_lit_bm_24,
                      load_Y2H_union_25,
                      load_AlphaFold_RoseTTAFold)

# %%
def bali2_connection(**kwargs):
    """Connect to DFCI CCSB bali2 database.

    Will ask for your username/password once and save it for future use.

    Arguments:
        **kwargs: passed to pymysql.connect

    Returns:
        pymysql connection

    """
    host = 'bali2.dfci.harvard.edu'
    config_dir = ccsblib_config_dir()
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    config_path = os.path.join(config_dir, 'config.txt')
    config = configparser.ConfigParser()
    con_fail_msg = ('Failed to connect to database.\nMaybe you are outside '
                    'the internal network and forgot to to start the VPN?')
    if os.path.exists(config_path):
        config.read(config_path)
        un = config['paros.dfci.harvard.edu']['username']
        pw = keyring.get_password('paros.dfci.harvard.edu', un)
        if pw is not None:
            try:
                return pymysql.connect(host=host, user=un, password=pw, **kwargs)
            except pymysql.OperationalError as e:
                if e.args[0] == 1045:
                    print('Wrong username or password')
                elif e.args[0] == 2003:
                    raise UserWarning(con_fail_msg)
                else:
                    raise e
    success = False
    while not success:
        try:
            un = input('paros username: ')
            pw = getpass.getpass('password: ')
            con = pymysql.connect(host, user=un, password=pw)
            success = True
            config[host] = {'username': un}
            with open(config_path, 'w') as f:
                config.write(f)
            try:
                keyring.set_password(host, un, pw)
            except RuntimeError:
                warnings.warn('WARNING: saving password failed')
        except pymysql.OperationalError as e:
            if e.args[0] == 1045:
                print('Wrong username or password')
            elif e.args[0] == 2003:
                raise UserWarning(con_fail_msg)
            else:
                raise e
    return con


# %%
def save_ensembl_to_uniprot_mapping():
    qry = """
            SELECT uniprot_accession AS uniprot_ac,
                    id AS ensembl_gene_id
               FROM horfeome_annotation_gencode27.uniprot2ensemble_20170914
              WHERE category = 'Ensembl';
            """
    pd.read_sql(qry, bali2_connection()).to_csv(
        '../data/internal/ensembl_to_uniprot_mapping.tsv',
        sep='\t',
        index=False,
        na_rep='NULL')
    

save_ensembl_to_uniprot_mapping()

# %%
from ccsblib.huri import load_protein_coding_genome

pcg = load_protein_coding_genome()
with open("../data/internal/protein_coding_genome.txt", "w") as f:
    f.write('\n'.join(pcg))

# %%
from ccsblib.huri import load_nw_hi_union, map_nw_ids

def load_uniprot_proteome_from_fasta(tax_id):
    uniprot_proteome_ids = {
        9606: 'UP000005640',
        10090: 'UP000000589',
        6239: 'UP000001940',
        3702: 'UP000006548',
        83333: 'UP000000625',
        559292: 'UP000002311',
        2697049: 'UP000464024',
        7227: 'UP000000803'
    }
    with open('../data/external/uniprot_2022-03-02/{}_{}.fasta'.format(uniprot_proteome_ids[tax_id], tax_id), 'r') as f:
        proteins = {l.split('|')[1] for l in f.readlines() if l.startswith('>')}
    return proteins


valid_uniprot_acs = load_uniprot_proteome_from_fasta(9606)
hi = load_nw_hi_union(id_type='ensembl_gene_id')
hi.to_csv('../data/internal/HI_union_ensembl_gene_id.tsv', sep='\t', na_rep='NULL', index=True)
hi = map_nw_ids(hi, 'ensembl_gene_id', 'uniprot_ac')
hi = hi.loc[hi['uniprot_ac_a'].isin(valid_uniprot_acs) & hi['uniprot_ac_b'].isin(valid_uniprot_acs), :]
hi.to_csv('../data/internal/HI_union_uniprot_ac.tsv', sep='\t', na_rep='NULL', index=True)

# %%
from datasets import load_zhang_et_al_biorxiv


def load_y2h_afrf_human_pairwise_test():
    df = pd.read_csv('../data/internal/Y2H_v1_pairwise_test_AlphaFoldRoseTTAFold_human.tsv',
                    sep='\t')
    df['final_score'] = df['final_score'].fillna('NA')
    df['seq_confirmation_final_3at'] = df['seq_confirmation_final_3at'].astype(pd.BooleanDtype())
    df['seq_confirmation_final_lw'] = df['seq_confirmation_final_lw'].astype(pd.BooleanDtype())
    df['result'] = df['final_score'].map(
        {
            '0': False,
            '1': True,
            'AA': np.nan,
            'NA': np.nan,
        }
    )
    if ((df['result'] == False) & df['seq_confirmation_final_3at'].notnull()).any():
        raise UserWarning('Inconsistency')
    df['seq_confirmation_combined'] = (
        ((df['result'] == True) & (df['seq_confirmation_final_3at'] == True))
        | ((df['result'] == False) & (df['seq_confirmation_final_lw'] == True))
    )
    df.loc[df['seq_confirmation_combined'] == False, 'result'] = np.nan
    df = _add_prs_rrs_results(df)
    df['orf_id_pair'] = (df[['ad_orf_id', 'db_orf_id']].min(axis=1).astype(pd.Int64Dtype()).astype(str)
                    + '_' +
                df[['ad_orf_id', 'db_orf_id']].max(axis=1).astype(pd.Int64Dtype()).astype(str))
    df = _add_is_published_column(df)


    return df

def _add_prs_rrs_results(df):
    qry_prs = """
    SELECT db_orf_id, 
           ad_orf_id, 
           category,  
           a.final_score  
      FROM ppi_prediction_challenge.retest_scoring_pred2425 a,
           ppi_prediction_challenge.retest b    
     WHERE a.standard_batch  = 'HPRSRRSv2' 
       AND b.standard_batch  = 'HPRSRRSv2' 
       AND a.retest_pla = b.retest_pla 
       AND a.retest_pos = b.retest_pos ;
    """
    df_prs = pd.read_sql(qry_prs, bali2_connection())
    df_prs['result'] = df_prs['final_score'].map(
        {
            '0': False,
            '1': True,
            'AA': np.nan,
            'NA': np.nan,
        })
    df_prs['category'] = df_prs['category'].map({
        'PRS': 'hsPRS-v2',
        'RRS': 'hsRRS-v2',
    })
    df = pd.concat([df, df_prs], axis=0, ignore_index=True)
    if df['final_score'].isnull().any():
        raise UserWarning('unexpected missing values')
    return df


def _add_is_published_column(df):
    # pairs were originally randomly selected from the biorxiv version
    df['in_biorxiv_version'] = df['category'].map(lambda x: True if x == 'Zhang_et_al' else pd.NA)
    zhang_biorxiv = load_zhang_et_al_biorxiv()
    zhang_biorxiv = zhang_biorxiv.rename(
                        columns={'#Protein1': 'uniprot_ac_a',
                                 'Protein2': 'uniprot_ac_b'})
    zhang_biorxiv['uniprot_pair'] = (zhang_biorxiv['uniprot_ac_a'] 
                                     + '_' 
                                     + zhang_biorxiv['uniprot_ac_b'])
    zhang_published = pd.read_excel('../data/external/science.adt1630_data_s1_to_s8.xlsx', 
                    sheet_name='Data S3', skiprows=15)
    zhang_published = zhang_published.rename(
                        columns={'Protein1': 'uniprot_ac_a',
                                 'Protein2': 'uniprot_ac_b'})
    zhang_published['uniprot_pair'] = (zhang_published['uniprot_ac_a'] 
                                     + '_' 
                                     + zhang_published['uniprot_ac_b'])
    if not (zhang_published['uniprot_ac_a'] < zhang_published['uniprot_ac_b']).all():
        raise UserWarning('expected sorted')
    uniprot_to_orf_id = pd.read_csv('../data/internal/uniprot_ac_to_orf_id_used_for_Zhang_et_al_experiment.tsv',
                                    sep='\t')
    zhang_mapped = (pd.merge(zhang_biorxiv, 
            uniprot_to_orf_id,
            how='left',
            left_on='uniprot_ac_a',
            right_on='uniprot_ac')
    .rename(columns={'uniprot_ac': 'uniprot_ac_1',
                    'orf_id': 'orf_id_1'})
    )
    zhang_mapped = (pd.merge(zhang_mapped, 
            uniprot_to_orf_id,
            how='left',
            left_on='uniprot_ac_b',
            right_on='uniprot_ac')
    .rename(columns={'uniprot_ac': 'uniprot_ac_2',
                    'orf_id': 'orf_id_2'})
    )
    zhang_mapped['orf_id_pair'] = (
        zhang_mapped[['orf_id_1', 'orf_id_2']].min(axis=1).astype(pd.Int64Dtype()).astype(str)
        + '_' +
        zhang_mapped[['orf_id_1', 'orf_id_2']].max(axis=1).astype(pd.Int64Dtype()).astype(str)
    )

    df = pd.merge(
        df,
        zhang_mapped.loc[:, ['orf_id_pair', 'uniprot_ac_a', 'uniprot_ac_b', 'uniprot_pair']],
        how='left',
        on='orf_id_pair',
    )
    df['in_published_version'] = df['uniprot_pair'].isin(zhang_published['uniprot_pair'].values)
    df.loc[df['category'] != 'Zhang_et_al', 'in_published_version'] = pd.NA
    df = df.drop_duplicates(subset=['category', 'ad_orf_id', 'db_orf_id'])
    

    orf_id_to_uniprot = uniprot_to_orf_id.drop_duplicates(subset=['orf_id']).set_index('orf_id')['uniprot_ac'].to_dict()
    qry_prs_uniprot = """select orf_ida as orf_id, uniprot_a as uniprot_ac
    from prsrrsv2.prsrrsv2
    union
    select orf_idb as orf_id, uniprot_b as uniprot_ac
    from prsrrsv2.prsrrsv2;"""
    for _i, row in pd.read_sql(qry_prs_uniprot, bali2_connection()).iterrows():
        orf_id_to_uniprot[row['orf_id']] = str(row['uniprot_ac']).split('-')[0]
    # hand adding three missing
    orf_id_to_uniprot[6318] = 'P62805'
    orf_id_to_uniprot[14083] = 'P62805'
    orf_id_to_uniprot[14085] = 'P0C0S8'

    df['uniprot_ac_a'] = df['uniprot_ac_a'].fillna(
        pd.concat(
            [df['ad_orf_id'].map(orf_id_to_uniprot),
             df['db_orf_id'].map(orf_id_to_uniprot)],
            axis=1
        ).min(axis=1)
    )
    df['uniprot_ac_b'] = df['uniprot_ac_b'].fillna(
        pd.concat(
            [df['ad_orf_id'].map(orf_id_to_uniprot),
             df['db_orf_id'].map(orf_id_to_uniprot)],
            axis=1
        ).max(axis=1)
    )

    if (df['uniprot_ac_a'] > df['uniprot_ac_b']).any():
        raise UserWarning('expected sorted')
    df = df.drop(columns=['uniprot_pair'])

    save_supplementary_table_Y2H_human(df)

    # random PRS drop
    # shuffle to take random orientation because 
    # PRS tested in both DB-X AD-Y and DB-Y AD-X
    # but other pairs tested in a single orientation
    df = (
        df.sample(frac=1, replace=False, random_state=3842467597)  # random shuffle
          .assign(has_result=df["result"].notna())
          .sort_values("has_result", ascending=False)
          .drop_duplicates(["category", "orf_id_pair"], keep="first")
          .drop(columns="has_result")
    )

    df = df.loc[(df['in_published_version'] == True)
                | (df['category'] != 'Zhang_et_al'), :]
    
    df.to_csv('../data/processed/Y2H_v1_pairwise_test_AlphaFoldRoseTTAFold_human_filtered.tsv',
              sep='\t',
              index=False,
              na_rep='NULL')


def save_supplementary_table_Y2H_human(df):
    tb = df.copy()

    tb['source_dataset'] = tb.category.apply(lambda x: {'PRS': 'hsPRS-v2', 'RRS': 'hsRRS-v2'}.get(x, x))
    tb = tb.rename(columns={
        'category': 'source_dataset',
        'ad_orf_id': 'AD_CCSB_ORF_ID',
        'db_orf_id': 'DB_CCSB_ORF_ID',
    })

    tb['result'] = tb['result'].map({True: 'Positive', False: 'Negative', np.nan: 'Test failed'})
    tb.loc[tb['final_score'] == 'AA', 'result'] = 'Autoactivator'
    tb.loc[(tb['seq_confirmation_combined'] == False) & (tb['result'] == 'Test failed'), 'result'] = 'Failed sequence confirmation'
    tb.loc[tb['final_score'] == 'NA', 'result'] = 'Test failed'
    tb = tb.loc[:, ['uniprot_ac_a', 
                    'uniprot_ac_b',
                    'AD_CCSB_ORF_ID',
                    'DB_CCSB_ORF_ID',
                    'source_dataset', 
                    'result',
                    'in_biorxiv_version',
                    'in_published_version'
                    ]]
    tb.to_csv('../supplementary_tables/Y2H_v1_pairwise_test_AlphaFoldRoseTTAFold_human.tsv', 
              index=False, 
              sep='\t',
              na_rep='NULL')


df = load_y2h_afrf_human_pairwise_test()

# %%
from datasets import load_yi_i

def save_all_curated_PPI_evidence_yeast_24():
    valid_orfs = load_non_dubious_orfs()
    qry = """SELECT *
            FROM bioinfo_lukel.lit24_evidence_yeast;"""
    evid = pd.read_sql(qry, bali2_connection())
    evid = evid.rename(columns={'sgd_orf_id_a': 'orf_name_a',
                                'sgd_orf_id_b': 'orf_name_b'})
    evid = evid.loc[evid['orf_name_a'] != evid['orf_name_b'], :]
    evid = evid.loc[evid['orf_name_a'].isin(valid_orfs) &
                    evid['orf_name_b'].isin(valid_orfs), :]
    
    pmid_date = load_pubmed_to_date_mapping('../data/processed/pmid_dates_yeast.csv')
    evid['date'] = evid['pubmed_id'].map(pmid_date)

    # add Yu et al back in
    yi1 = load_yi_i(remove_homodimers=True)
    yi1['n_pairs_per_experiment'] = yi1.shape[0]
    yi1['binary'] = 1
    yi1['y2h'] = 1
    yi1['method_id'] = 'MI:0232'
    yi1['date'] = pd.to_datetime('2008-10-03')
    if yi1.index.name != 'pair':
        raise UserWarning('expected index name to be "pair"')
    evid = pd.concat([evid, yi1.reset_index()], axis=0)
    evid['pair'] = evid['orf_name_a'] + '_' + evid['orf_name_b']
    evid.to_csv('../data/internal/all_curated_PPI_evidence_yeast_24.tsv',
                sep='\t',
                index=False)
    
save_all_curated_PPI_evidence_yeast_24()

# %%
# cut pasted from human timeline notebook
organism = 'human'
qry_pairs = """SELECT *
            FROM bioinfo_lukel.lit24_pairs_{};""".format(organism)
qry_evid = """SELECT *
            FROM bioinfo_lukel.lit24_evidence_{};""".format(organism)
conn = bali2_connection()
lit = pd.read_sql(qry_pairs, conn)
evid = pd.read_sql(qry_evid, conn)
lit.to_csv('../data/internal/Lit-24-human.tsv', 
           sep='\t',
           na_rep='NULL', 
           index=False)
evid.to_csv('../data/internal/Lit-24-human_evidence.tsv', 
           sep='\t',
           na_rep='NULL', 
           index=False)

# %%
def save_ge_pcc_values():
    con = bali2_connection()
    qry_ge = """select CONCAT(orf_name_a, '_', orf_name_b) AS pair,
                orf_name_a,
                orf_name_b,
                `value` as PCC
                from yi2_paper.coxpresdb_full;"""
    ge = pd.read_sql(qry_ge, con)
    ge["PCC"] = ge["pair"].map(ge.groupby("pair")["PCC"].mean())
    ge = ge.drop_duplicates().set_index("pair")
    ge.to_csv('../data/processed/GE_PCC_values.tsv', 
              sep='\t',
              index=True,
              na_rep='NULL')
    
save_ge_pcc_values()

# %%
# cut pasted from literature filtering notebook
import pandas as pd
from orangecontrib.bio import ontology

from utils import bali2_connection


def load_ito_core():
    df = pd.read_csv('../data/external/Ito_et_al_PNAS_2001_core_only.txt', 
                       sep='\t',
                       names=['a', 'b'])
    df['sgd_orf_id_a'] = df[['a', 'b']].min(axis=1)
    df['sgd_orf_id_b'] = df[['a', 'b']].max(axis=1)
    df = df.drop(columns=['a', 'b'])
    df['pair'] = df['sgd_orf_id_a'] + '_' + df['sgd_orf_id_b']
    df = df.set_index('pair')
    return df


def load_uetz_library():
    df = pd.read_csv('../data/external/Uetz_et_al_Nature_2000_Table2_screen_only.txt', 
                     sep='\t', 
                     names=['sgd_orf_id_a', 'sgd_orf_id_b'])
    df['pair'] = df['sgd_orf_id_a'] + '_' + df['sgd_orf_id_b']
    df = df.set_index('pair')
    return df

# TODO: turn into function
pubmed_ids_to_remove = {
    18467557,  # Tarassov
    11805826,  # Gavin 2002
    16429126,  # Gavin 2006
    16554755,  # Krogan
    11805837,  # Ho
    19095691,  # CYC2008 -- No pairs
    17200106,  # Collins -- No pairs
    }
ito_pmid = 11283351
uetz_pmid = 10688190
ito_core = load_ito_core()
uetz_screen = load_uetz_library()

qry = """SELECT *
           FROM bioinfo_lukel.lit24_evidence_yeast;"""
evid = pd.read_sql(qry, bali2_connection())

evid = evid.loc[~evid['pubmed_id'].isin(pubmed_ids_to_remove), :]

evid_pair = (evid[['sgd_orf_id_a', 'sgd_orf_id_b']].min(axis=1) +
             '_' +
             evid[['sgd_orf_id_a', 'sgd_orf_id_b']].max(axis=1))
if evid_pair.isnull().any():
    raise UserWarning('unexpected null values')

evid = evid.loc[~(evid_pair.isin(uetz_screen.index) & (evid['pubmed_id'] == uetz_pmid)), :]
evid = evid.loc[~(evid_pair.isin(ito_core.index) & (evid['pubmed_id'] == ito_pmid)), :]

a = evid[['sgd_orf_id_a', 'sgd_orf_id_b']].min(axis=1)
b = evid[['sgd_orf_id_a', 'sgd_orf_id_b']].max(axis=1)
evid['sgd_orf_id_a'] = a
evid['sgd_orf_id_b'] = b

evid = evid.drop_duplicates(['sgd_orf_id_a', 'sgd_orf_id_b', 'pubmed_id', 'method_id'])

def categorize_pairs(evid, miOntology):
    """Divide the literature into Lit-BM/BS etc.
    Args:
        lit (DataFrame): table of literature evidences.
        miOntolgoy (OBOOntology): PSI-MI molecular interaction (MI) ontology
    Returns:
        DataFrame: table of pairs of proteins.
    """
    lit = evid.copy()
    y2hTerms = set(map(lambda x: x.id,
                       miOntology.sub_terms('MI:0232')))
    y2hTerms.add('MI:0232')
    lit['y2h'] = lit['method_id'].isin(y2hTerms)
    lit['binary_not_y2h'] = lit['binary'] & (~lit['y2h'])
    pairs = (lit.groupby(['sgd_orf_id_a', 'sgd_orf_id_b'])
                .agg({'binary_not_y2h': 'any',
                      'binary': 'any',
                      'pubmed_id': 'nunique',
                      'method_id': 'nunique'})
                .rename(columns={'pubmed_id': 'n_publications',
                                 'method_id': 'n_methods'}))
    pairs['n_evidences'] = lit.groupby(['sgd_orf_id_a', 'sgd_orf_id_b']).size()
    pairs['category'] = 'Lit-NB'
    pairs.loc[pairs['binary'] & (pairs['n_evidences'] >= 2), 'category'] = 'Lit-BM'
    pairs.loc[pairs['binary'] & (pairs['n_evidences'] == 1), 'category'] = 'Lit-BS'
    pairs = pairs.reset_index()
    return pairs


miOntology = ontology.OBOOntology('../data/external/psi-mi.obo')
lit = categorize_pairs(evid, miOntology)

lit.to_csv('../data/internal/Lit-24-yeast.tsv',
           sep='\t',
           index=False)
evid.to_csv('../data/internal/Lit-24-yeast_evidence.tsv',
            sep='\t',
            index=False)

# %%
def _add_gene_names(df_in):
    if 'orf_name_a' not in df_in.columns or 'orf_name_b' not in df_in.columns:
        raise ValueError()
    df = df_in.copy()
    gene_names = load_all_orfs().set_index('orf_name')['gene_name']
    df['gene_name_a'] = df['orf_name_a'].map(gene_names)
    df['gene_name_b'] = df['orf_name_b'].map(gene_names)
    name_columns = ['orf_name_a', 'gene_name_a', 'orf_name_b', 'gene_name_b']
    df = df.loc[:, name_columns + [c for c in df.columns if c not in name_columns]]
    return df

# %%
def load_Y2H_version_benchmarking():
    df = pd.read_csv('../data/internal/scPRSv2_scRRSv2_rescored.tsv',
                 sep='\t')
    df.loc[(df['S07'] == '1') &
           (df['seq_S07'] != 'confirmed'), 'S07'] = 'sequence_confirmation_failed'
    df.loc[(df['r02 v1'] == '1') &
           (df['seq_r02_v1'] != 'confirmed'), 'r02 v1'] = 'sequence_confirmation_failed'
    df.loc[(df['r02 v4'] == '1') &
           (df['seq_r02_v4'] != 'confirmed'), 'r02 v4'] = 'sequence_confirmation_failed'
    df.loc[:, ['r02 v1', 'r02 v4', 'S07']] = df.loc[:, ['r02 v1', 'r02 v4', 'S07']].applymap(lambda x: {
                                                'sequence_confirmation_failed': 'Failed sequence confirmation',
                                                     '0': 'Negative',
                                                     '1': 'Positive',
                                                     'NAN': 'Test failed',
                                                     'AA': 'Autoactivator',
                                                     np.nan: 'Test failed'}.get(x, x))
    df = df.drop(columns=['S07', 'seq_S07', 'seq_r02_v1', 'seq_r02_v4'])
    df = df.rename(columns={'r02 v1': 'Y2H_v1', 'r02 v4': 'Y2H_v4'})
    return df

# %%
def load_CCSB_YI_II_search_space():
    """
    
    """
    qry = """SELECT orf_name, accession, orf_space, in_db_space, in_ad_space
               FROM yi2_paper.yi2_space
              WHERE in_yeri_paper_space = 1;"""
    df = pd.read_sql(qry, bali2_connection())
    df = df.rename(columns={'accession': 'GenBank_accession'})
    df['orf_space'] = df['orf_space'].map({'II': 'FLEXGene', 'III': 'CCSB_additional'})
    df['in_db_space'] = df['in_db_space'].astype(bool)
    df['in_ad_space'] = df['in_ad_space'].astype(bool)
    df = df.rename(columns={'orf_space': 'ORFeome_collection',
                            'in_db_space': 'screened_as_DB',
                            'in_ad_space': 'screened_as_AD'})
    if df['orf_name'].duplicated().any():
        raise UserWarning('Unexpected duplicates')
    if df['GenBank_accession'].dropna().duplicated().any():
        raise UserWarning('Unexpected duplicates')
    return df

# %%
def load_CCSB_YI_II():
    qry = """select ad_orf_name, db_orf_name, space, 
                    in_screen_1, in_screen_2, in_screen_3,
                    manual_score_growth
              from yi2_paper.yi2_final;"""
    df = pd.read_sql(qry, bali2_connection())
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['ad_orf_name'].isin(valid_orfs) &
                df['db_orf_name'].isin(valid_orfs), :]
    df = df.drop(columns=['space'])
    return df

# %%
def load_ABBI():
    qry = """SELECT orf_name_a,
                    orf_name_b,
                    dataset
               FROM yi2_paper.dataset_final
              WHERE dataset in ('YI2', 'CCSB-YI1', 'ITO_CORE', 'UETZ_SCREEN');"""
    df = pd.read_sql(qry, bali2_connection())
    df = pd.concat([df.loc[:, ['orf_name_a', 'orf_name_b']],
                    pd.get_dummies(df['dataset'])], axis=1)
    df = df.groupby(['orf_name_a', 'orf_name_b']).any().reset_index()
    df['pair'] = df['orf_name_a'] + '_' + df['orf_name_b']
    df = df.set_index('pair')
    # gene names
    sgd_orfs = load_all_orfs().drop_duplicates('orf_name').set_index('orf_name')
    df = pd.merge(df, sgd_orfs, how='left', left_on='orf_name_a', right_index=True, suffixes=('', '_a'))
    df = pd.merge(df, sgd_orfs, how='left', left_on='orf_name_b', right_index=True, suffixes=('_a', '_b'))
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_a'].isin(valid_orfs) &
                    df['orf_name_b'].isin(valid_orfs), :]
    df = df.rename(columns={'UETZ_SCREEN': 'Uetz-screen',
                                'ITO_CORE': 'Ito-core',
                                'YI2': 'YeRI'})
    df = df.loc[:, ['orf_name_a', 'gene_name_a', 'orf_name_b', 'gene_name_b',
                        'Uetz-screen', 'Ito-core', 'CCSB-YI1', 'YeRI']]
    return df

# %%
def load_GPCA_and_MAPPIT_data():
    qry = """SELECT *
    FROM
    (
    SELECT a.test_orf_ida as orf_id_a,
           a.test_orf_idb as orf_id_b,
           a.test_pla, a.test_pos,
           a.assay, 
           a.standard_batch, 
           a.final_score, 
           a.final_score_id,
           b.source
      FROM yeri_validation.validation AS a
     INNER JOIN yeri_validation.validation_source AS b
        ON a.standard_batch = b.standard_batch
       AND a.test_orf_ida = b.orf_id1
       AND a.test_orf_idb = b.orf_id2

    UNION

    SELECT c.test_orf_ida as orf_id_a,
           c.test_orf_idb as orf_id_b,
           c.test_pla, c.test_pos,
           c.assay, 
           c.standard_batch, 
           c.final_score, 
           c.final_score_id,
           d.source
      FROM yeri_validation.validation AS c
     INNER JOIN yeri_validation.validation_source AS d
        ON c.standard_batch = d.standard_batch
       AND c.test_orf_ida = d.orf_id2
       AND c.test_orf_idb = d.orf_id1
     WHERE c.test_orf_ida != c.test_orf_idb
    ) AS e

    WHERE standard_batch != 'Sv05'
    ;"""
    df = pd.read_sql(qry, bali2_connection())
    df.groupby(['standard_batch', 'source']).size()
    df['pair'] = (df[['orf_id_a', 'orf_id_b']].min(axis=1).astype(str) + '_' + 
                  df[['orf_id_a', 'orf_id_b']].max(axis=1).astype(str))
    
    
    source_rename = {'1screen0.3': 'YeRI',
                 '1screen0.3_r02': 'YeRI',
                 '2screens': 'YeRI',
                 '2screens_r02': 'YeRI',
                 '3screens': 'YeRI',
                 '3screens_r02': 'YeRI',
                     'YI34': 'YeRI',
                     'Double_edge': 'YeRI',
                 'CYC-2008-2-50': 'CYC2008',
                 'Uetz-et-al-screen': 'Uetz-screen',
                 'Ito-et-al-core': 'Ito-core',
                 'Tarrassov-et-al': 'Tarassov',
                 'YI1': 'CCSB-YI1',
                 'prs': 'scPRS-v2',
                 'prs_Sv01_config': 'scPRS-v2',
                 'rrs': 'scRRS-v2',
                 'rrs2': 'scRRS-v2',
                 'Additional_emptyAD_pairs': 'YeRI',
                 'Screen': 'YeRI',
                 'Lit-BM-13': 'Lit-BM-13',
                     'SGA-profiling-02': 'GI_PSN_PCC_gt_0.2',
                     'SGA-profiling-03': 'GI_PSN_PCC_gt_0.3',
                     'SGA-profiling-05': 'GI_PSN_PCC_gt_0.5',
                     'LitBS': 'Lit-BS-17',
                     'LitBM2': 'Lit-BM-17_2_evidence',
                     'LitBM3': 'Lit-BM-17_3_evidence',
                     'LitBM4': 'Lit-BM-17_4+_evidence'
                }
    source_rename.update({'Redo_failed_' + k: v for k, v in source_rename.items()})
    df['source'] = df['source'].map(lambda x: source_rename.get(x, x))
    
    #to_drop = {'SGA-genetic-interm', 'Rpd3L', 'Rpd3S', 'Protogene', 'Double_edge', 'rrs_proto', 'Collins-et-al'}
    to_drop = {'SGA-genetic-interm', 'Rpd3L', 'Rpd3S', 'Protogene', 'rrs_proto', 'Collins-et-al'}
    df = df.loc[~df['source'].isin(to_drop), :]
    
    # there are some pairs of PRS tested in both orientations in Sv06
    # keep just the orientation that is the same as Sv01 
    prs_sv01 = (df.loc[(df['standard_batch'] == 'Sv01') & (df['source'] == 'scPRS-v2'),
                        'orf_id_a'].astype(str) + '-' +
                df.loc[(df['standard_batch'] == 'Sv01') & (df['source'] == 'scPRS-v2'),
                        'orf_id_b'].astype(str))
    df['is_prs_sv01_orientation'] = ~(df['orf_id_a'].astype(str) + '-' + df['orf_id_b'].astype(str)).isin(prs_sv01)
    df = df.sort_values(['standard_batch', 'pair', 'source', 'is_prs_sv01_orientation'])
    df = df.drop_duplicates(['pair', 'standard_batch', 'source'])
    
    dna_conc = load_GPCA_DNA_concentration()
    df = pd.merge(df, dna_conc.loc[dna_conc['test_config'] == 'A',
                              ['test_pla', 'test_pos', 'standard_batch', 'orf_id', 'DNA_concentration']]
                         .rename(columns={'orf_id': 'orf_id_a',
                                          'DNA_concentration': 'DNA_concentration_a'}),
             how='left',
             on=['orf_id_a', 'standard_batch', 'test_pla', 'test_pos'])
    df = pd.merge(df, dna_conc.loc[dna_conc['test_config'] == 'B',
                              ['test_pla', 'test_pos', 'standard_batch', 'orf_id', 'DNA_concentration']]
                         .rename(columns={'orf_id': 'orf_id_b',
                                          'DNA_concentration': 'DNA_concentration_b'}),
             how='left',
             on=['orf_id_b', 'standard_batch', 'test_pla', 'test_pos'])
    df['DNA_concentration_min'] = df[['DNA_concentration_a', 'DNA_concentration_b']].min(axis=1)
    if ((df['assay'] == 'GPCA') & df['DNA_concentration_min'].isnull()).any():
        raise UserWarning('Missing DNA concentration values for GPCA')
    
    df.loc[(df['DNA_concentration_min'] < 25), 'final_score'] = np.nan
    
    orf_map = load_orf_id_map()
    df['orf_name_a'] = df.apply(lambda x: min([orf_map[x['orf_id_a']], orf_map[x['orf_id_b']]]),
                                  axis=1)
    df['orf_name_b'] = df.apply(lambda x: max([orf_map[x['orf_id_a']], orf_map[x['orf_id_b']]]),
                                  axis=1)
    df['pair'] = df['orf_name_a'] + '_' + df['orf_name_b']
    df['orf_name_f1'] = df['orf_id_a'].map(orf_map)
    df['orf_name_f2'] = df['orf_id_b'].map(orf_map)
    
    # remove PRS/RRS that are not in v2 and YI2 pairs no longer in dataset
    scPRSv2 = load_scPRSv2_pairs()
    scRRSv2 = load_scRRSv2_pairs()
    abbi = load_ABBI()
    yi2 = set(abbi[abbi['YeRI']].index.values)
    old_yi2 = (df['source'] == 'YeRI') & (~df['pair'].isin(yi2))
    old_prs = (df['source'] == 'scPRS-v2') & (~df['pair'].isin(scPRSv2))
    old_rrs = (df['source'] == 'scRRS-v2') & (~df['pair'].isin(scRRSv2))
    df = df.loc[~(old_prs | old_rrs | old_yi2), :]
    

    df = df.rename(columns={'standard_batch': 'experiment_ID',
                            'source': 'source_dataset',
                            'final_score': 'score'})
    
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_f1'].isin(valid_orfs) &
                df['orf_name_f2'].isin(valid_orfs),
                :]
    
    # remove YeRI pairs tested twice
    yi2_tested = {}
    for exp_id in df['experiment_ID'].unique():
        yi2_tested[exp_id] = set(df.loc[(df['experiment_ID'] == exp_id) &
                                        (df['source_dataset'] == 'YeRI') &
                                        df['score'].notnull(),
                                        'pair'].values)
    repeats = (
                (
                  (df['experiment_ID'] == 'Sv02') &
                  (df['source_dataset'] == 'YeRI') & 
                  df['pair'].isin(yi2_tested['Sv01'])
                )
                |
                (
                  (df['experiment_ID'] == 'Sv06') &
                  (df['source_dataset'] == 'YeRI') & 
                  df['pair'].isin(yi2_tested['Sv01'].union(yi2_tested['Sv02']))
                )
                |
                (
                  (df['experiment_ID'] == 'Sv04') &
                  (df['source_dataset'] == 'YeRI') & 
                  df['pair'].isin(yi2_tested['Sv03'])
                )
              )
    df = df.loc[~repeats, :]
    
    df['F1_CCSB_ORF_ID'] = df['orf_id_a']
    df['F2_CCSB_ORF_ID'] = df['orf_id_b']
    df = df.loc[:, ['assay',
                    'experiment_ID',
                    'orf_name_a',
                    'orf_name_b',
                    'orf_name_f1',
                    'orf_name_f2',
                    'F1_CCSB_ORF_ID',
                    'F2_CCSB_ORF_ID',
                    'source_dataset',
                    'score']]
    
    df['result_at_0_RRS'] = np.nan
    for std_batch in df['experiment_ID'].unique():
        exp = (df['experiment_ID'] == std_batch)
        nonan = df['score'].notnull()
        cutoff = df.loc[exp & (df['source_dataset'] == 'scRRS-v2'), 'score'].max()
        df.loc[exp & nonan, 'result_at_0_RRS'] = (df.loc[exp & nonan, 'score'] > cutoff)

    return df


def load_orf_id_map():
    qry = """select orf_id, orf_name
              from yeast_interactome_ii.master_ref;"""
    
    df = pd.read_sql(qry, bali2_connection())
    if df['orf_id'].duplicated().any():
        raise UserWarning('Unexpected duplicates')
    return df.set_index('orf_id')['orf_name'].to_dict()


def load_GPCA_DNA_concentration():    
    
    def convert_wrong_position_strings(s):
        """In the excel files, positions are e.g. D2 instead of D02"""
        return s[0] + s[1:].zfill(2)

    GPCA_DNA_data_dir = Path('../data/from_alice/GPCA_DNA')
    ############### Sv01 ####################
    conc_a = pd.read_excel(GPCA_DNA_data_dir / 'CP_dilutions_N1_N2.xlsx',
                           sheet_name='N1',
                          usecols=['plate', 'well', 'DNA_concentration'],
                         dtype={'plate': str, 
                                   'well': str, 
                                    'DNA_concentration': np.float64})
    conc_b = pd.read_excel(GPCA_DNA_data_dir / 'CP_dilutions_N1_N2.xlsx',
                           sheet_name='N2',
                          usecols=['plate', 'well', 'DNA_concentration'],
                          dtype={'plate': str, 
                                   'well': str, 
                                    'DNA_concentration': np.float64})
    conc_a['well'] = conc_a['well'].apply(convert_wrong_position_strings)
    conc_b['well'] = conc_b['well'].apply(convert_wrong_position_strings)
    conc_a['plate'] = conc_a['plate'].apply(lambda x: x.split('_')[0] + '_' + {'N1': 'Ea', 'N2': 'Eb'}[x.split('_')[1]])
    conc_b['plate'] = conc_b['plate'].apply(lambda x: x.split('_')[0] + '_' + {'N1': 'Ea', 'N2': 'Eb'}[x.split('_')[1]])
    # THE ORF ID HERE IS A STRING!!!!!
    # ARRRRGGGGGHHHHHHHH
    qry = """SELECT a.orf_id, a.test_config, a.node_plate, a.node_pos,
                    b.dstplate AS test_plate, b.dstpos as test_pos
              FROM yeri_validation.validation_nodes AS a
             INNER JOIN yeri_validation.validation_round2_cherrymap AS b
              ON REPLACE(REPLACE(a.node_plate, 'Ea', 'N1'),'Eb', 'N2') = b.srcplate
             AND a.node_pos = b.srcpos; ;"""
    node_plate_map = pd.read_sql(qry, bali2_connection())
    node_plate_map['orf_id'] = node_plate_map['orf_id'].apply(lambda x: int(x.split('_')[0]))

    df = pd.merge(pd.concat([conc_a, conc_b]),
                  node_plate_map,
                             how='inner',
                             left_on=['plate', 'well'],
                             right_on=['node_plate', 'node_pos']
                    )
    df['standard_batch'] = 'Sv01'
    df = df.loc[:, ['standard_batch', 'orf_id', 'test_config', 'test_plate', 'test_pos', 'DNA_concentration']]

    ########## Sv02 ###################
    conc_a = pd.concat([pd.read_excel(GPCA_DNA_data_dir / 'N1 dilutions for CP.xlsx',
                                      usecols=['plate', 'well', 'DNA_concentration'],
                                dtype={'plate': str, 
                                   'well': str, 
                                    'DNA_concentration': np.float64}),
                       pd.read_excel(GPCA_DNA_data_dir / 'CP_Dilution_PRS_RSS.xlsx',
                                     sheet_name='N1',
                                     usecols=['plate', 'well', 'DNA_concentration'],
                                    dtype={'plate': str, 
                                   'well': str, 
                                    'DNA_concentration': np.float64})])
    conc_b = pd.concat([pd.read_excel(GPCA_DNA_data_dir / 'N2 dilutions for CP.xlsx',
                                      usecols=['plate', 'well', 'DNA_concentration']),
                        pd.read_excel(GPCA_DNA_data_dir / 'CP_Dilution_PRS_RSS.xlsx',
                                      sheet_name='N2',
                                      usecols=['plate', 'well', 'DNA_concentration'])])
    conc_a['well'] = conc_a['well'].apply(convert_wrong_position_strings)
    conc_b['well'] = conc_b['well'].apply(convert_wrong_position_strings)
    conc_a['plate'] = conc_a['plate'].apply(lambda x: x.split('_')[0])
    conc_b['plate'] = conc_b['plate'].apply(lambda x: x.split('_')[0])
    qry = """SELECT test_orf_ida, test_orf_idb, test_pla_full AS test_plate, test_pos
               FROM yeri_validation.validation
              WHERE standard_batch = 'Sv02';"""
    plate_map = pd.read_sql(qry, bali2_connection())
    df = pd.concat([df, 
        (pd.merge(conc_a,
                         plate_map,
                         how='inner',
                         left_on=['plate', 'well'],
                         right_on=['test_plate', 'test_pos'])
                  .assign(standard_batch='Sv02',
                          test_config='A')
                  .rename(columns={'test_orf_ida': 'orf_id'})
                  .loc[:, ['standard_batch', 'orf_id', 'test_config', 'test_plate', 'test_pos', 'DNA_concentration']]),
        (pd.merge(conc_b,
                         plate_map,
                         how='inner',
                         left_on=['plate', 'well'],
                         right_on=['test_plate', 'test_pos'])
                  .assign(standard_batch='Sv02',
                          test_config='B')
                  .rename(columns={'test_orf_idb': 'orf_id'})
                  .loc[:, ['standard_batch', 'orf_id', 'test_config', 'test_plate', 'test_pos', 'DNA_concentration']])
                   ])
    df['test_plate'] = df['test_plate'].str.replace('_N[1,2]', '')
    
    ########### Sv06 ###########################
    conc_a = pd.read_excel(GPCA_DNA_data_dir / 'Sv06 Master Sheet updated v2.xlsx',
                                     sheet_name='Ea',
                                     usecols=['plate', 'well', 'DNA_concentration'],
                            dtype={'plate': str, 
                                   'well': str, 
                                    'DNA_concentration': np.float64})
    conc_b = pd.read_excel(GPCA_DNA_data_dir / 'Sv06 Master Sheet updated v2.xlsx',
                                      sheet_name='Eb',
                                      usecols=['plate', 'well', 'DNA_concentration'],
                            dtype={'plate': str, 
                                   'well': str, 
                                    'DNA_concentration': np.float64})
    conc_a['well'] = conc_a['well'].apply(convert_wrong_position_strings)
    conc_b['well'] = conc_b['well'].apply(convert_wrong_position_strings)
    # weird inconsistent naming of plates
    conc_a['plate'] = conc_a['plate'].apply(lambda x: x.split('_')[0] if len(x.split('_')) == 2 else x[:-3].replace('_', 'G'))
    conc_b['plate'] = conc_b['plate'].apply(lambda x: x.split('_')[0] if len(x.split('_')) == 2 else x[:-3].replace('_', 'G'))
    qry = """SELECT test_orf_ida, test_orf_idb, test_pla_full AS test_plate, test_pos
               FROM yeri_validation.validation
              WHERE standard_batch = 'Sv06';"""
    plate_map = pd.read_sql(qry, bali2_connection())
    plate_map['test_plate'] = plate_map['test_plate'].apply(lambda x: x.split('_')[0])
    df = pd.concat([df, 
               (pd.merge(conc_a,
                         plate_map,
                         how='inner',
                         left_on=['plate', 'well'],
                         right_on=['test_plate', 'test_pos'])
                  .assign(standard_batch='Sv06',
                          test_config='A')
                  .rename(columns={'test_orf_ida': 'orf_id'})
                  .loc[:, ['standard_batch', 'orf_id', 'test_config', 'test_plate', 'test_pos', 'DNA_concentration']]),
        (pd.merge(conc_b,
                         plate_map,
                         how='inner',
                         left_on=['plate', 'well'],
                         right_on=['test_plate', 'test_pos'])
                  .assign(standard_batch='Sv06',
                          test_config='B')
                  .rename(columns={'test_orf_idb': 'orf_id'})
                  .loc[:, ['standard_batch', 'orf_id', 'test_config', 'test_plate', 'test_pos', 'DNA_concentration']])
                   ])
    
    df = df.rename(columns={'test_plate': 'test_pla'})
    df['test_pla'] = df['test_pla'].apply(lambda x: int(x[-3:]))
    
    # There are a small number of duplicates readings. Asked Alice and she said to take the
    # highest, since the duplicates were repeat measurements after the machine failed.
    df = df.groupby(['standard_batch', 'orf_id', 'test_config', 'test_pla', 'test_pos'])['DNA_concentration'].max().reset_index()
    
    if df.isnull().any().any():
        raise UserWarning('Unexpected null values')
    return df


def load_scPRSv2_pairs():
    qry = """select f_orf_name, 
                     r_orf_name
    from yi2_paper.yi2_prsrrs_plate_map
    where f_orf_name is not NULL
    and cat_final = 'PRS';"""
    df = pd.read_sql(qry, bali2_connection())
    return set((df[['r_orf_name', 'f_orf_name']].min(axis=1) + '_' +
                df[['r_orf_name', 'f_orf_name']].max(axis=1)).values)


def load_scRRSv2_pairs():
    qry = """select f_orf_name, 
                    r_orf_name
    from yi2_paper.yi2_prsrrs_plate_map
    where f_orf_name is not NULL
    and cat_final = 'RRS';"""
    df = pd.read_sql(qry, bali2_connection())
    return set((df[['r_orf_name', 'f_orf_name']].min(axis=1) + '_' +
                df[['r_orf_name', 'f_orf_name']].max(axis=1)).values)

def load_scPRSv2():
    qry = """select f_orf_name, 
                     r_orf_name
    from yi2_paper.yi2_prsrrs_plate_map
    where f_orf_name is not NULL
    and cat_final = 'PRS';"""
    df = pd.read_sql(qry, bali2_connection())
    df['orf_name_a'] = df[['r_orf_name', 'f_orf_name']].min(axis=1)
    df['orf_name_b'] = df[['r_orf_name', 'f_orf_name']].max(axis=1)
    df = df.drop(columns=['r_orf_name', 'f_orf_name'])
    df = df.drop_duplicates()
    return df


def load_scRRSv2():
    qry = """select f_orf_name, 
                    r_orf_name
    from yi2_paper.yi2_prsrrs_plate_map
    where f_orf_name is not NULL
    and cat_final = 'RRS';"""
    df = pd.read_sql(qry, bali2_connection())
    df['orf_name_a'] = df[['r_orf_name', 'f_orf_name']].min(axis=1)
    df['orf_name_b'] = df[['r_orf_name', 'f_orf_name']].max(axis=1)
    df = df.drop(columns=['r_orf_name', 'f_orf_name'])
    df = df.drop_duplicates()
    return df

# %%
# TMP
df = load_GPCA_and_MAPPIT_data()
# TMP
df.to_csv('../supplementary_tables/GPCA_and_MAPPIT_data.tsv',
          sep='\t',
          index=False)
df.to_csv('../data/internal/GPCA_and_MAPPIT_data.tsv',
          sep='\t',
          index=False)

# %%
def load_Y2H_pairwise_test():
    qry = """SELECT orf_name_a,
                    orf_name_b,
                    ad_orf_name,
                    db_orf_name,
                    ad_orf_id as AD_CCSB_ORF_ID,
                    db_orf_id as DB_CCSB_ORF_ID,
                    final_call,
                    seq_confirmation,
                    dataset AS source_dataset
               FROM yi2_paper.validation_paper_final
              INNER JOIN yi2_paper.retest_source
              USING (orf_name_a, orf_name_b)
              WHERE assay = 'S07';"""
    df = pd.read_sql(qry, bali2_connection())
    
    df['result'] = np.nan
    df.loc[(df['final_call'] == '1') &
           (df['seq_confirmation'] != 'y'), 'final_call'] = 'seq_fail'
    df['result'] = df['final_call'].map({'seq_fail': 'Failed sequence confirmation',
                                         '0': 'Negative',
                                         '1': 'Positive',
                                         'NA': 'Test failed',
                                         'AA': 'Autoactivator'})
    if df['result'].isnull().any():
        raise UserWarning('Problem with result column')
    df = df.drop(columns=['final_call', 'seq_confirmation'])
    
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_a'].isin(valid_orfs) &
                  df['orf_name_b'].isin(valid_orfs), :]
    
    # Fix mistake in DB where dataset = " PRS" for one pair
    df['source_dataset'] = df['source_dataset'].str.strip()
    
    unused_data = {'DEL_PCC_0.2', 'DEL_PCC_0.5', 'DEL_PCC_0.7', 'BABU', 'MARCOTTE'}
    df = df.loc[~df['source_dataset'].isin(unused_data), :]
    rename = {'PRS': 'scPRS-v2',
              'RRS': 'scRRS-v2',
              'LIT_BM_17': 'Lit-BM-17',
              'LIT_BS_17': 'Lit-BS-17',
              'LIT_NB_17': 'Lit-NB-17',
              'TARASSOV': 'Tarassov',
              'PREPPI': 'PrePPI-LR600',
              'YI1': 'CCSB-YI1',
              'YI2': 'YeRI',
              'UETZ_SCREEN': 'Uetz-screen',
              'ITO_CORE': 'Ito-core',
              'INTERACTOME3D': 'I3D-exp-17',
              'GI_PCC_0.2': 'GI_PSN_PCC_gt_0.2',
              'GI_PCC_0.3': 'GI_PSN_PCC_gt_0.3',
              'GI_PCC_0.5': 'GI_PSN_PCC_gt_0.5',
              'HO': 'Ho',
              'KROGAN': 'Krogan',
              'GAVIN_2002': 'Gavin_2002',
              'GAVIN_2006': 'Gavin_2006',
              'GERSTEIN': 'Jansen',
              'CYC2008': 'CYC2008'}
    df['source_dataset'] = df['source_dataset'].map(rename)
    if df['source_dataset'].isnull().any():
        raise UserWarning('something wrong with dataset name mapping')
    
    prs = load_scPRSv2_pairs()
    rrs = load_scRRSv2_pairs()
    df['pair'] = df['orf_name_a'] + '_' + df['orf_name_b']
    df = df.loc[~((df['source_dataset'] == 'scPRS-v2') & 
                  ~df['pair'].isin(prs)), :]
    df = df.loc[~((df['source_dataset'] == 'scRRS-v2') & 
                  ~df['pair'].isin(rrs)), :]
    df = df.drop(columns=['pair'])
    return df

# %%
def load_alphafold_Y2H_pairwise_test():
    qry = """
        SELECT a.ad_orf_name,
               a.db_orf_name,
               a.ad_orf_id AS AD_CCSB_ORF_ID,
               a.db_orf_id AS DB_CCSB_ORF_ID,
               source AS source_dataset,
               final_score, seq_confirmation
          FROM yeri_retest.retest_source  AS a,
               yeri_retest.retest AS b 
         WHERE a.standard_batch  = 'YS08'  
           AND a.standard_batch = b.standard_batch
           AND a.ad_orf_id = b.ad_orf_id 
           AND a.db_orf_id = b.db_orf_id;
        """
    df = pd.read_sql(qry, bali2_connection())
    df['result'] = np.nan
    df.loc[(df['final_score'] == '1') &
           (df['seq_confirmation'] != 'y'), 'final_score'] = 'seq_fail'
    df['result'] = df['final_score'].map({'seq_fail': 'Failed sequence confirmation',
                                         '0': 'Negative',
                                         '1': 'Positive',
                                         'NA': 'Test failed',
                                         'AA': 'Autoactivator'})
    if df['result'].isnull().any():
        raise UserWarning('Problem with result column')
    df['source_dataset'] = df['source_dataset'].map({'PRS': 'scPRS-v2',
                                                     'RRS': 'scRRS-v2',
                                                     'alpha_no_overlap': 'AlphaFold+RoseTTAFold'})
    df = df.drop(columns=['final_score', 'seq_confirmation'])

    # only keep PRS/RRS in a single orientation
    qry_ori = """
        SELECT c.ad_orf_name, c.db_orf_name  
        FROM  yi2_paper.validation_paper_final AS a,
            yeri_retest.retest AS b,
            yeri_retest.retest_source AS c 
        WHERE b.ad_orf_id = c.ad_orf_id 
        AND b.db_orf_id = c.db_orf_id 
        AND a.final_score_id = b.final_score_id 
        AND assay= 'S07'
        AND b.standard_batch = c.standard_batch  
        AND (source  LIKE '%PRS%' OR source LIKE '%RRS%');"""
    ori = pd.read_sql(qry_ori, bali2_connection())
    ori['same_orientation_as_previous_experiment'] = True
    df = pd.merge(df, ori,  how='left')
    df.loc[df['source_dataset'].isin(['scPRS-v2', 'scRRS-v2']) &
           df['same_orientation_as_previous_experiment'].isnull(),
           'same_orientation_as_previous_experiment'] = False
    df['orf_name_a'] = df[['ad_orf_name', 'db_orf_name']].min(axis=1)
    df['orf_name_b'] = df[['ad_orf_name', 'db_orf_name']].max(axis=1)
    df = df[['orf_name_a', 'orf_name_b'] + list(df.columns[:-2])]

    return df

load_alphafold_Y2H_pairwise_test().to_csv(
    '../data/internal/Y2H_v4_pairwise_test_AlphaFoldRoseTTAFold.tsv',
    sep='\t', 
    index=False)
load_alphafold_Y2H_pairwise_test().to_csv(
    '../supplementary_tables/Y2H_v4_pairwise_test_AlphaFoldRoseTTAFold.tsv',
    sep='\t', 
    index=False)

# %%
# TMP
from datasets import load_additional_y2h_pairwise_test

df = load_additional_y2h_pairwise_test(remove_homodimers=True)
df.duplicated(subset=['orf_name_a', 'orf_name_b', 'source_dataset']).sum()

# %%
df.head()

# %%
def load_Lit_13():
    qry = """SELECT orf_name_a,
                    orf_name_b,
                    n_evidences AS n_evidence,
                    category
               FROM yi2_paper.lit_13_final;"""
    df = pd.read_sql(qry, bali2_connection())
    df['category'] = 'Lit-' + df['category']
    df = _add_gene_names(df) 
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_a'].isin(valid_orfs) &
                df['orf_name_b'].isin(valid_orfs), :]
    return df

# %%
def load_Lit_17():
    qry = """SELECT orf_name_a,
                    orf_name_b,
                    publications AS n_publications,
                    n_methods,
                    n_evidences AS n_evidence,
                    category
               FROM yi2_paper.lit_17_final;"""
    df = pd.read_sql(qry, bali2_connection())
    df = _add_gene_names(df)
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_a'].isin(valid_orfs) &
                df['orf_name_b'].isin(valid_orfs), :]
    return df

# %%
def load_Lit_24():
    """
    
    This is the literature curated PPI data, with proteome-wide systematic maps removed.
    Those maps are: Ito-core, Uetz-library, YI-I (Yu et al.), Tarassov et al. Gavin et al. 2002
    and 2006, Ho and Krogan.
    
    """
    df = pd.read_csv('../data/internal/Lit-24-yeast.tsv', sep='\t')
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_a'].isin(valid_orfs) &
                df['orf_name_b'].isin(valid_orfs), :]
    df = df.drop(columns=['binary_not_y2h', 'binary'])
    df = _add_gene_names(df)
    return df

# %%
def load_Lit_24_evidence():
    df = pd.read_csv('../data/internal/Lit-24-yeast_evidence.tsv', sep='\t')
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df['orf_name_a'].isin(valid_orfs) &
                df['orf_name_b'].isin(valid_orfs), :]
    df = df.loc[:, ['orf_name_a',
                    'orf_name_b',
                    'pubmed_id',
                    'method_id',
                    'method_name',
                    'binary',
                    'y2h',
                    'n_pairs_per_experiment']]
    df = _add_gene_names(df)
    return df

# %%
def load_direct_and_indirect_pairs():

    df = pd.concat([pd.read_csv('../data/from_carles/yeast_{}_pairs.min_complex_size_3.tsv'.format(s), sep='\t') 
                    for s in ['direct', 'indirect']],
                    ignore_index=True)
    df = df.drop(columns=['pair_ID'])
    df = df.rename(columns={'protein1': 'uniprot_ac_a',
                            'protein2': 'uniprot_ac_b'})
    return df

# %%
import datasets
from datasets import load_I3D_exp_20, load_PDB_metadata


def load_interface_area():
    evid = pd.read_csv('../data/external/interactome3d_2020-05_scer_interactions_complete.dat', sep='\t')
    pdb_metadata = load_PDB_metadata(evid['PDB_ID'].unique())
    interface_area = datasets.load_interface_area()
    i3d = load_I3D_exp_20(remove_homodimers=True)
    i3d['interface_area'] = i3d.index.map(interface_area)
    for column in pdb_metadata.columns:
        i3d[column] = i3d['PDB_ID'].map(pdb_metadata[column])
    i3d['min_coverage'] = i3d[['COVERAGE1', 'COVERAGE2']].min(axis=1)
    i3d['mean_coverage'] = i3d[['COVERAGE1', 'COVERAGE2']].mean(axis=1)

    n_contacts = (pd.read_csv('../data/processed/n_residue_contacts_interactome3d_2020_01.txt',
                            sep=r'\s+',
                            names=['structure', 'n_residue_contacts'])
    .set_index('structure')
    ['n_residue_contacts']
    .to_dict()
    )
    i3d['n_residue_contacts'] = (i3d['FILENAME']
                                .apply(lambda x: x.split('.')[0])
                                .apply(lambda x: n_contacts.get(x, np.nan))
    )
    old_index = i3d.index.copy()
    i3d = pd.merge(i3d,
            evid.groupby(['PROT1', 'PROT2'])['PDB_ID'].nunique().rename('n_structures').reset_index(),
            how='left',
            left_on=['uniprot_ac_1', 'uniprot_ac_2'],
            right_on=['PROT1', 'PROT2'])
    i3d.index = old_index


    energy_i3d = pd.read_csv('../data/internal/foldx_AC_optimized_repaired_I3D-exp-20.txt', sep='\t')
    energy_i3d['filename_short'] = energy_i3d['Pdb'].str.slice(len('./Optimized_'), -len('_Repair.pdb'))
    i3d['filename_short'] = i3d['FILENAME'].apply(lambda x: x.split('.')[0])
    i3d = pd.merge(i3d,
                    energy_i3d,
                    on=['filename_short'],
                    how='left')
    i3d.index = old_index

    i3d = i3d.loc[:, ['orf_name_a', 'orf_name_b', 'PDB_ID', 'interface_area', 
                      'date', 'n_polypeptide',
       'n_dna', 'n_rna', 'n_sugar', 'n_ligand', 'n_carbohydrate_polymer',
       'experimental_method', 'polymeric_count', 'min_coverage',
       'mean_coverage', 'n_residue_contacts', 'IntraclashesGroup1',
       'IntraclashesGroup2', 'Interaction Energy', 'StabilityGroup1',
       'StabilityGroup2']]

    return i3d

# %%
# TODO: add all this in to function below
df = pd.read_csv('../supplementary_tables/functional_predictions.tsv', sep='\t')
df['dataset'] = df['dataset'].map({'YeRI': 'YeRI', 'ABBI-21': 'Y2H-union-25'})
df = df.rename(columns={
    'ORF_ID': 'orf_name',
    'GO_size_in_PPI': 'n_proteins_with_GO_term_in_network',
    'raw_effect': 'n_interactors_with_GO_term',
    'random_effect': 'mean_in_random_networks',
    'random_std': 'std_in_random_networks',
    })
df['n_interactors_with_GO_term'] = df['n_interactors_with_GO_term'].astype(int)

go_dag = GODag('../data/external/go.obo', load_obsolete=True)
df['GO_term_name'] = df['GO_ID'].map(lambda x: go_dag[x].name if x in go_dag else None)
if df['GO_term_name'].isnull().any():
    raise UserWarning("Some GO terms don't have names")
df['GO_term_name'] = df['GO_term_name'].str.replace('obsolete ', '')
(df.loc[(df['z-score'] >= 5) & (df['n_interactors_with_GO_term'] >= 2)]
 .to_csv('../supplementary_tables/functional_predictions_filtered.tsv', sep='\t', index=False))


def load_istvans_functional_predictions():
    data_dir = Path('../data/from_istvan')
    with open(data_dir / 'filt_header.dat', 'r') as f:
        header = f.read().split('\t')
    with open(data_dir / 'GO_mapping.dat', 'r') as f:
        go_map = {int(l.strip().split('\t')[1]): l.strip().split('\t')[0] for l in f.readlines()}
    with open(data_dir / 'protein_mapping_v2.dat', 'r') as f:
        orf_map = {int(l.strip().split('\t')[1]): l.strip().split('\t')[0] for l in f.readlines()}
    old_names = ['yi2', 'pre', 'post', 'lit2']
    new_names  = ['YeRI', 'Y2H-union', 'ABBI-21', 'Lit-BM-17']
    tables = []
    for old_name, new_name in zip(old_names, new_names):
        for method in ['d', 'w']:
            df = pd.read_csv(data_dir / 'filt_{}_{}.dat'.format(old_name, method),
                             sep='\t',
                             names=header)
            df['dataset'] = new_name
            df['method'] = {'d': 'first_neighbors', 'w': 'second_neighbors'}[method]
            tables.append(df)

    df = pd.concat(tables)
    df['GO_ID'] = df['GO_ID'].map(go_map)
    df['Protein_ID'] = df['Protein_ID'].map(orf_map)
    df = df.rename(columns={'Protein_ID': 'ORF_ID'})
    
    # In the manuscript we just use ABBI-21, YeRI and first neighbors
    df = df.loc[df['dataset'].isin({'ABBI-21', 'YeRI'}) &
                (df['method'] == 'first_neighbors'), :]
    df = df.drop(columns=['method', 'Protein_degree'])
    
    if df.isnull().any().any():
        raise UserWarning('Unexpected missing values')
    return df

# %%
def load_protein_properties():
    qry = """SELECT *
               FROM yi2_paper.samogram_rank;"""
    df = pd.read_sql(qry, bali2_connection())
    df = df.drop(columns=['orf_name_no_dash', 'pfam_coverage', 'go_term_30', 'iupred_idr', 'length'])
    variable_names = {
                  'pubmed_cnt': 'Publication count',
                  'abundance': 'Abundance',
                  'conservation_score': 'Conservation',
                  'essential': 'Essential'}
    df = df.rename(columns=variable_names)
    return df

# %%
from datasets import load_i3d_exp_17, load_I3D_exp_24

def load_all_other_networks():
    qry = """SELECT orf_name_a, orf_name_b, dataset
               FROM yi2_paper.dataset_final
              WHERE in_yeri_paper_space = 1
                AND dataset in ('GI_0.01',
                                'COXPRESdb_0.01',
                                 'Chemogenomics_Hom_0.01',
                                  'KROGAN',
                                  'GAVIN_2002',
                                   'GAVIN_2006',
                                    'HO',
                                    'TARASSOV',
                                     'GERSTEIN',
                                     'PREPPI',
                                     'CYC2008',
                                     'ITO_CORE',
                                     'UETZ_SCREEN',
                                     'CCSB-YI1');"""
    df = pd.read_sql(qry, bali2_connection())
    df['dataset'] = df['dataset'].map({'GI_0.01': 'GI-PSN',
                                       'COXPRESdb_0.01': 'GE-PSN',
                                       'Chemogenomics_Hom_0.01': 'CS-PSN',
                                       'KROGAN': 'Krogan',
                                       'GAVIN_2002': 'Gavin_2002',
                                       'GAVIN_2006': 'Gavin_2006',
                                       'CYC2008': 'CYC2008',
                                       'HO': 'Ho',
                                       'TARASSOV': 'Tarassov',
                                       'GERSTEIN': 'Jansen',
                                       'PREPPI': 'PrePPI-LR600',
                                       'ITO_CORE': 'Ito-core',
                                     'UETZ_SCREEN': 'Uetz-screen',
                                     'CCSB-YI1': 'CCSB-YI1',
                                       })

    df = pd.concat([
        df,
        load_i3d_exp_17(remove_homodimers=False)
          .loc[:, ['orf_name_a', 'orf_name_b']].assign(dataset='I3D-exp-17'),
                  load_I3D_exp_24(remove_homodimers=False)
          .loc[:, ['orf_name_a', 'orf_name_b']]
          .assign(dataset='I3D-exp-24')
    ])
    df = _add_gene_names(df)
    return df

# %%
def load_CCSB_YI1_space():
    qry = """SELECT * FROM yeast_interactome_ccsb.SPACE;"""
    df = pd.read_sql(qry, bali2_connection())
    df = pd.concat([df[['ORF_NAME']], pd.get_dummies(df['TYPE'])], axis=1)
    df = ((df.groupby('ORF_NAME').sum() > 0)
            .reset_index()
            .rename(columns={'ORF_NAME': 'orf_name',
                            'AD': 'screened_as_AD',
                            'DB': 'screened_as_DB'}))
    return df

# %%
def read_residue_contacts(in_path):
    with open(in_path, 'r') as f:
        # check format
        lines = f.readlines()
    if lines[0] != '    Residue     Atom   Residue     Atom  Type     Dist\n':
        raise UserWarning('unexpected format')
    table_per_residue = [l.strip().split() for l in lines[:-7]]
    table_per_residue = pd.DataFrame(columns=['Residue_X', 'Residue_index_X', 'Atom_X', 'Residue_Y', 'Residue_index_Y', 'Atom_Y', 'Type', 'Dist'], 
                                     data=table_per_residue[1:] if len(table_per_residue) > 1 else [])
    table_summary = [(l.split(':')[0], int(l.split()[-1].strip())) for l in lines[-6:]]
    table_summary = pd.DataFrame(data=table_summary).set_index(0).T
    table_per_residue['Residue_index_X'] = table_per_residue['Residue_index_X'].astype(int)
    table_per_residue['Residue_index_Y'] = table_per_residue['Residue_index_Y'].astype(int)
    return (table_per_residue, table_summary)

aa_seqs = {s.description.split()[0]: str(s.seq)[:-1] for s in SeqIO.parse('../data/external/orf_trans_all.fasta', 'fasta')}
prot_length = {k: len(v) for k, v in aa_seqs.items()}

def load_alphafold_residue_contacts(in_dir):
    full_table = []
    summary_table = []
    for fpath in in_dir.glob('*/ranked_0_residue_contacts_Interactome3D.out'):
        pair = fpath.parents[0].stem
        gene_name_a, gene_name_b  = pair.split('_')
        len_a = prot_length[gene_name_a]
        len_b = prot_length[gene_name_b]
        df_a, df_b = read_residue_contacts(fpath)
        # NOTE: this is 0-indexed and the two proteins are concatenated
        pae = pd.read_csv(fpath.parents[0] / 'ranked_0_PAE.csv', index_col=0)
        pae.columns = pae.columns.astype(int)

        # DEBUG
        #tmp = df_a.apply(lambda row: pae.at[row['Residue_index_X'] - 1, (len_a + row['Residue_index_Y']) - 1], axis=1)
        #print(tmp)

        if df_a.shape[0] > 0:
            df_a['PAE'] = df_a.apply(lambda row: pae.at[row['Residue_index_X'] - 1, (len_a + row['Residue_index_Y']) - 1], axis=1)
            df_a['pair'] = pair
            df_a['Chain_X'] = 'A'
            df_a['Chain_Y'] = 'B'
        df_b['pair'] = pair
        df_b['Chain_X'] = 'A'
        df_b['Chain_Y'] = 'B'
        full_table.append(df_a.copy())
        summary_table.append(df_b.copy())
    full_table = pd.concat(full_table)
    summary_table = pd.concat(summary_table)

    full_table['gene_name_a'] = full_table['pair'].apply(lambda x: x.split('_')[0])
    full_table['gene_name_b'] = full_table['pair'].apply(lambda x: x.split('_')[1])
    summary_table['gene_name_a'] = summary_table['pair'].apply(lambda x: x.split('_')[0])
    summary_table['gene_name_b'] = summary_table['pair'].apply(lambda x: x.split('_')[1])

    full_table = full_table.loc[:, list(full_table.columns[-5:]) + list(full_table.columns[:-5])]
    summary_table = summary_table.loc[:, list(summary_table.columns[-5:]) + list(summary_table.columns[:-5])]
    return full_table, summary_table



def read_pdockq(in_dir):
    data = []
    for fpath in in_dir.glob('*/ranked_0_pdockq.out'):
        with open(fpath, 'r') as f:
            lines = f.readlines()
        data.append((lines[0].split()[-1].split('/')[-2],
                    float(lines[0].split()[2]),
                    float(lines[1].split()[-1]),
        ))
    df = pd.DataFrame(data=data, columns=['pair', 'pDockQ', 'PPV_from_pDockQ'])
    return df

from alphafold_multimer_utils import read_ipae, read_iptm_plus_ptm
from datasets import load_I3D_exp_20

def load_AF_results_PRS():
    in_dirs = [Path('../data/alphafold/scRRSv2/'), Path('../data/alphafold/scPRSv2/')]
    dfs = []
    for in_dir in in_dirs:
        data = []
        for fpath in in_dir.glob('*/ranked_0_contact_probability.txt'):
            with open(fpath, 'r') as f:
                data.append((fpath.parents[0].stem, float(f.read())))
        df = pd.DataFrame(data=data, columns=['pair', 'contact_probability'])
        df = pd.merge(df,
                    read_pdockq(in_dir),
                    on=['pair'],
                    how='left')
        df = pd.merge(df,
                    read_ipae(in_dir),
                    on=['pair'],
                    how='left')
        df = pd.merge(df,
                    read_iptm_plus_ptm(in_dir),
                    on=['pair'],
                    how='left')
        full_table, summary_table = load_alphafold_residue_contacts(in_dir)
        for pae_cutoff in [4,]:
            n_conf_rrc = full_table.loc[full_table['PAE'] <= pae_cutoff, :].groupby('pair').size()
            df[f'# residue contacts PAE ≤ {pae_cutoff} Å'] = df['pair'].map(n_conf_rrc).fillna(0)
        df['source'] = in_dir.stem
        dfs.append(df.copy())
    df = pd.concat(dfs)

    df['orf_name_a'] = df['pair'].apply(lambda x: x.split('_')[0])
    df['orf_name_b'] = df['pair'].apply(lambda x: x.split('_')[1])
    orfs = load_all_orfs()
    df['gene_name_a'] = df['orf_name_a'].map(orfs.set_index('orf_name')['gene_name'])
    df['gene_name_b'] = df['orf_name_b'].map(orfs.set_index('orf_name')['gene_name'])
    df['gene_name_a'] = df['gene_name_a'].fillna(df['orf_name_a'])
    df['gene_name_b'] = df['gene_name_b'].fillna(df['orf_name_b'])
    df['is_homodimer'] = (df['orf_name_a'] == df['orf_name_b'])
    df = df.loc[~df['is_homodimer'], :].drop(columns=['is_homodimer'])  # 

    i3d = load_I3D_exp_20(remove_homodimers=False)
    for other_dataset, name in [(i3d, 'I3D-exp-20'),]:
        df['in_' + name] = df['pair'].isin(other_dataset.index)

    # interologs
    def load_full_interactome_3d():  
        id_map = pd.read_csv(
            "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
        )
        id_map[3] = id_map[3].fillna(id_map[1])
        id_map = id_map.loc[:, [1, 3, 6]]
        id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]  
        i3d = pd.read_csv('../data/external/interactome3d_2020-05_scer_interactions.dat',
                        sep='\t')
        i3d = pd.merge(i3d, id_map, how="inner", left_on="PROT1", right_on="uniprot_ac")
        i3d = pd.merge(
            i3d,
            id_map,
            how="inner",
            left_on="PROT2",
            right_on="uniprot_ac",
            suffixes=("_1", "_2"),
        )
        i3d["orf_name_a"] = i3d[["orf_name_1", "orf_name_2"]].min(axis=1)
        i3d["orf_name_b"] = i3d[["orf_name_1", "orf_name_2"]].max(axis=1)
        i3d["pair"] = i3d["orf_name_a"] + "_" + i3d["orf_name_b"]
        i3d = i3d.drop_duplicates("pair")
        if i3d["pair"].duplicated().any():
            raise UserWarning("Unexpected duplicate pairs")
        i3d = i3d.set_index("pair").drop(columns=["PROT1", "PROT2"])
        return i3d

    full_i3d = load_full_interactome_3d()
    df = df.rename(columns={'in_I3D-exp-20': 'has_experimental_structure',
                            'iPTM+PTM': 'model_confidence'})
    df['has_homologous_structure'] = df['pair'].isin(full_i3d.loc[full_i3d['TYPE'] != 'Structure'].index)
    df = df.reset_index(drop=True)

    af3_dir = Path('../data/alphafold/AF3_scPRSv2_scRRSv2')
    af3_results = []
    for subdir in af3_dir.iterdir():
        if not subdir.stem.startswith('fold_'):
            continue
        with open(subdir / (subdir.stem + '_job_request.json'), 'r') as f:
            pair = json.load(f)[0]["name"]
        with open(subdir / (subdir.stem + '_summary_confidences_0.json'), 'r') as f:
            conf_scores = json.load(f)
            iptm = conf_scores["iptm"]
            ptm = conf_scores["ptm"]
        af3_results.append({'pair': pair, 'AF3_iPTM': iptm, 'AF3_PTM': ptm})
    af3_results = pd.DataFrame(af3_results)
    af3_results['AF3_model_confidence'] = 0.8 * af3_results['AF3_iPTM'] + 0.2 * af3_results['AF3_PTM']
    df = pd.merge(df, af3_results, how='left', on=['pair'])

    return df

# %%
from alphafold_multimer_utils import read_alphafold_dimer_metrics
from datasets import (load_all_orfs,
                      load_I3D_exp_20,
                      load_lit_24,
                      load_AlphaFold_RoseTTAFold,
                      load_Y2H_union_25,
                      load_tarassov,
                      load_sys_nb,
                      load_gi_pcc_values,
                      load_cs_pcc_values,
                      load_ge_pcc_values,
                      load_non_dubious_orfs,
                      )




def load_full_interactome_3d():  
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map[3] = id_map[3].fillna(id_map[1])
    id_map = id_map.loc[:, [1, 3, 6]]
    id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]  
    i3d = pd.read_csv('../data/external/interactome3d_2020-05_scer_interactions.dat',
                      sep='\t')
    i3d = pd.merge(i3d, id_map, how="inner", left_on="PROT1", right_on="uniprot_ac")
    i3d = pd.merge(
        i3d,
        id_map,
        how="inner",
        left_on="PROT2",
        right_on="uniprot_ac",
        suffixes=("_1", "_2"),
    )
    i3d["orf_name_a"] = i3d[["orf_name_1", "orf_name_2"]].min(axis=1)
    i3d["orf_name_b"] = i3d[["orf_name_1", "orf_name_2"]].max(axis=1)
    i3d["pair"] = i3d["orf_name_a"] + "_" + i3d["orf_name_b"]
    i3d = i3d.drop_duplicates("pair")
    if i3d["pair"].duplicated().any():
        raise UserWarning("Unexpected duplicate pairs")
    i3d = i3d.set_index("pair").drop(columns=["PROT1", "PROT2"])
    return i3d


def load_AF_results_YeRI():

    apms = pd.read_csv('../data/external/Michaelis_et_al_Nature_2023_Edges.csv',
                    sep=';')
    if apms[['source', 'target']].isnull().any().any():
        raise UserWarning('unexpected missing values')
    apms["orf_name_a"] = apms[['source', 'target']].min(axis=1)
    apms["orf_name_b"] = apms[['source', 'target']].max(axis=1)
    apms["pair"] = apms["orf_name_a"] + "_" + apms["orf_name_b"]
    # some of the targets are have multiple ORFs that map to them
    apms["target"] = apms["target"].str.split(";")
    apms = apms.explode(column="target").reset_index(drop=True)
    # some of the targets and sources are dubious or transposible element genes
    valid_orfs = load_non_dubious_orfs()
    apms = apms.loc[apms["orf_name_a"].isin(valid_orfs)
                    & apms["orf_name_b"].isin(valid_orfs), :]
    if apms["pair"].duplicated().any():
        raise UserWarning('unexpected duplicates')
    apms = apms.set_index("pair")

    full_i3d = load_full_interactome_3d()

    df = read_alphafold_dimer_metrics('../data/alphafold/YeRI/')

    df['orf_name_a'] = df['pair'].apply(lambda x: x.split('_')[0])
    df['orf_name_b'] = df['pair'].apply(lambda x: x.split('_')[1])
    orfs = load_all_orfs()
    df['gene_name_a'] = df['orf_name_a'].map(orfs.set_index('orf_name')['gene_name'])
    df['gene_name_b'] = df['orf_name_b'].map(orfs.set_index('orf_name')['gene_name'])
    df['gene_name_a'] = df['gene_name_a'].fillna(df['orf_name_a'])
    df['gene_name_b'] = df['gene_name_b'].fillna(df['orf_name_b'])
    df['is_homodimer'] = (df['orf_name_a'] == df['orf_name_b'])
    df = df.loc[~df['is_homodimer'], :].drop(columns=['is_homodimer'])  # 

    val = datasets.load_GPCA_and_MAPPIT_data(remove_homodimers=False)
    df['in_MAPPIT'] = df['pair'].isin(val.loc[(val['assay'] == 'MAPPIT')
                                            & (val['result_at_0_RRS'] == True), 'pair'].values)
    df['in_GPCA'] = df['pair'].isin(val.loc[(val['assay'] == 'GPCA')
                                            & (val['result_at_0_RRS'] == True), 'pair'].values)

    i3d = load_I3D_exp_20(remove_homodimers=False)
    afrf = load_AlphaFold_RoseTTAFold(remove_homodimers=True, restrict_to_high_confidence=False)
    lit = load_lit_24(remove_homodimers=False)
    litbm = lit.loc[lit['category'] == 'Lit-BM', :].copy()
    litbs = lit.loc[lit['category'] == 'Lit-BS', :].copy()
    litnb = lit.loc[lit['category'] == 'Lit-NB', :].copy()
    tarassov = load_tarassov(remove_homodimers=False)
    sysnb = load_sys_nb(remove_homodimers=False)
    abbi = load_Y2H_union_25(remove_homodimers=True)
    yeri = abbi.loc[abbi['YeRI'], :].copy()
    y2h_union = abbi.loc[abbi['Uetz-screen'] & abbi['Ito-core'] & abbi['CCSB-YI1'] , :].copy()

    for other_dataset, name in [(i3d, 'I3D-exp-20'),
                    (afrf, 'AlphaFoldRoseTTAFold'),
                    (litbm, 'Lit-BM-24'),
                    (y2h_union, 'Y2H-union'),
                    (litbs, 'Lit-BS-24'),
                    (litnb, 'Lit-CC-24'),
                    (sysnb, 'AP/MS-06'),
                    (apms, 'AP/MS-23'),
                    (tarassov, 'Tarassov'),
                    ]:
        df['in_' + name] = df['pair'].isin(other_dataset.index)
    df['has_interolog_structure'] = df['pair'].isin(full_i3d.loc[full_i3d['TYPE'] != 'Structure'].index)

    df['in_AYC_pre_YeRI'] = (df['in_Lit-BM-24'] | 
                    df['in_Y2H-union'] | 
                    df['in_AlphaFoldRoseTTAFold'] |
                    df['in_I3D-exp-20'])
    df['novel'] = ~(df['in_AYC_pre_YeRI']
                    | df['in_AP/MS-06']
                    | df['in_AP/MS-23']
                    | df['in_Lit-BS-24']
                    | df['in_Lit-CC-24']
                    | df['in_Tarassov']
                    )

    df['no_previous_structure'] = ~(df['in_I3D-exp-20'] | df['in_AlphaFoldRoseTTAFold'] | df['has_interolog_structure'])

    df = df.rename(columns={'iPTM+PTM': 'model_confidence',
                            'in_MAPPIT': 'MAPPIT_result',
                            'in_GPCA': 'GPCA_result'})
    df = df.drop(columns=[c for c in df.columns if c.startswith('in_')])
    df = df.drop(columns=['has_interolog_structure'])

    return df

# %%
def load_AlphaFold_RoseTTAFold_interface_size():
    df = pd.read_csv('../data/internal/Interface-area-and-energy_Humphreys-Science-2021.tsv', sep='\t')
    # TMP -- replacing deltaG with updated but incomplete
    energy = pd.read_csv('../data/internal/foldx_AC_optimized_repaired_humphreys.txt', sep='\t')
    energy['filename'] = energy['Pdb'].str.slice(len('./Optimized_'), -len('_Repair.pdb'))
    df = pd.merge(df.iloc[:, :4], energy, how='left', on='filename')
    df = df.loc[:, ['gene1', 'gene2', 'filename', 'interface_area', 'IntraclashesGroup1', 'IntraclashesGroup2',
       'Interaction Energy', 'StabilityGroup1', 'StabilityGroup2']]
    return df

# %%
def load_high_quality_union():
    i3d = load_I3D_exp_24(remove_homodimers=False)
    lit = load_lit_bm_24(remove_homodimers=False)
    abbi = load_Y2H_union_25(remove_homodimers=False)
    afrf = load_AlphaFold_RoseTTAFold(remove_homodimers=True,
                                        restrict_to_high_confidence=True)
    df = pd.concat(
        [i3d.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
        lit.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
                    abbi.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
                    afrf.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy()]
        )
    df = df.drop_duplicates('pair').set_index('pair')
    df = _add_gene_names(df)
    df['in_I3D-exp-24'] = df.index.isin(i3d.index)
    df['in_Lit-BM-24'] = df.index.isin(lit.index)
    df['in_AFRF-core'] = df.index.isin(afrf.index)
    df['in_ABBI-24'] = df.index.isin(abbi.index)
    return df

# %%
def load_valbin24():
    i3d = load_I3D_exp_24(remove_homodimers=False)
    lit = load_lit_bm_24(remove_homodimers=False)
    y2h_union = load_Y2H_union_25(remove_homodimers=False)
    y2h_union = y2h_union.loc[y2h_union['Ito-core'] |
                            y2h_union['Uetz-screen'] |
                            y2h_union['CCSB-YI1']]
    df = pd.concat(
        [i3d.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
            lit.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
            y2h_union.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
        ]
        )
    df = df.drop_duplicates('pair').set_index('pair')
    df = _add_gene_names(df)
    df['in_I3D-exp-24'] = df.index.isin(i3d.index)
    df['in_Lit-BM-24'] = df.index.isin(lit.index)
    df['in_Y2H-union-08'] = df.index.isin(y2h_union.index)
    return df

# %%
def load_valbin25():
    i3d = load_I3D_exp_24(remove_homodimers=False)
    lit = load_lit_bm_24(remove_homodimers=False)
    y2h_union = load_Y2H_union_25(remove_homodimers=False)
    df = pd.concat(
        [i3d.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
            lit.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
            y2h_union.loc[:, ['orf_name_a', 'orf_name_b']].reset_index().copy(),
        ]
        )
    df = df.drop_duplicates('pair').set_index('pair')
    df = _add_gene_names(df)
    df['in_I3D-exp-24'] = df.index.isin(i3d.index)
    df['in_Lit-BM-24'] = df.index.isin(lit.index)
    df['in_Y2H-union-25'] = df.index.isin(y2h_union.index)
    return df

# %%
# TODO: 
# PRS/RRS needs mappit and gpca? the results are in the MAPPIT and GPCA table
# Lit-NB --> Lit-CC
tables = {
    'CCSB-YI1_space': load_CCSB_YI1_space,
    'protein_properties': load_protein_properties,
    'scPRS-v2': load_scPRSv2,
    'scRRS-v2': load_scRRSv2,
    'Y2H_assay_version_benchmarking': load_Y2H_version_benchmarking,
    'interface_surface_area': load_interface_area,
    'Lit-24-yeast': load_Lit_24,
    'Lit-24-yeast_evidence': load_Lit_24_evidence,
    'Lit-13-yeast': load_Lit_13,
    'Lit-17-yeast': load_Lit_17,
    'all_other_networks': load_all_other_networks,
    'Y2H_v4_pairwise_test': load_Y2H_pairwise_test,
    'GPCA_and_MAPPIT': load_GPCA_and_MAPPIT_data,
    'direct_vs_indirect_contacts_complex_size_gte_3': load_direct_and_indirect_pairs,
    'Y2H_v4_pairwise_test_AlphaFoldRoseTTAFold': load_alphafold_Y2H_pairwise_test,
    'AFRF_interface_size': load_AlphaFold_RoseTTAFold_interface_size,
    'ORF-search-space_YeRI': load_CCSB_YI_II_search_space,
    'YeRI': load_CCSB_YI_II,
    'Y2H-union-25': load_Y2H_union_25,
    'functional_predictions': load_istvans_functional_predictions,
    'AlphaFold_confidence_scPRS-v2_scRRS-v2': load_AF_results_PRS,
    'AlphaFold_confidence_YeRI': load_AF_results_YeRI,
    'high-quality-union': load_high_quality_union,
    'ValBin-24': load_valbin24,
    'ValBin-25': load_valbin25
          }
for name, function in tables.items():
    continue  # TMP remove after testing
    function().to_csv('../supplementary_tables/' + name + '.tsv',
                      sep='\t',
                      index=False)

# %%



