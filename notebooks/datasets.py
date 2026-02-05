import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import bioservices
from bioservices import KEGG


YERI_DATE = "2025-07-09"
AFRF_YEAST_PUB_DATE = "2021-11-11"


def load_non_dubious_orfs():
    df = load_all_orfs()
    return set(
        df.loc[df["qualifier"].isin(["Verified", "Uncharacterized"]), "orf_name"].values
    )


def load_all_orfs():
    """

    Note: 4 silenced ORFs have been removed from this file:
    YCR097W/HMRA1, YCR096C/HMRA2, YCL066W/HMLALPHA1, YCL067C/HMLALPHA2

    """
    df = pd.read_csv("../data/processed/SGD_protein_coding_ORFs_20170114.tsv", sep="\t")
    return df


def load_y2h_prs_rrs_data():
    df = pd.read_csv("../data/internal/scPRSv2_scRRSv2_rescored.tsv", sep="\t")
    df.loc[(df["S07"] == "1") & (df["seq_S07"] != "confirmed"), "S07"] = (
        "sequence_confirmation_failed"
    )
    df.loc[(df["r02 v1"] == "1") & (df["seq_r02_v1"] != "confirmed"), "r02 v1"] = (
        "sequence_confirmation_failed"
    )
    df.loc[(df["r02 v4"] == "1") & (df["seq_r02_v4"] != "confirmed"), "r02 v4"] = (
        "sequence_confirmation_failed"
    )
    df["pair_gene_names"] = df["gene_name_X"] + " " + df["gene_name_Y"]
    df.loc[:, ["r02 v1", "r02 v4", "S07"]] = df.loc[
        :, ["r02 v1", "r02 v4", "S07"]
    ].applymap(
        lambda x: {
            "sequence_confirmation_failed": "Failed sequence confirmation",
            "0": "Negative",
            "1": "Positive",
            "NAN": "Test failed",
            "AA": "Autoactivator",
            np.nan: "Test failed",
        }.get(x, x)
    )
    return df


def load_y2h_assay_version_benchmarking_experiment():
    df = pd.read_csv("../data/internal/Y2H_assay_version_benchmarking.tsv", sep="\t")
    df["pair_gene_names"] = df["gene_name_X"] + " " + df["gene_name_Y"]
    return df


def load_GPCA_and_MAPPIT_data(*, remove_homodimers):
    df = pd.read_csv("../data/internal/GPCA_and_MAPPIT.tsv", sep="\t")
    df["pair"] = (
        df[["orf_name_f1", "orf_name_f2"]].min(axis=1)
        + "_"
        + df[["orf_name_f1", "orf_name_f2"]].max(axis=1)
    )
    if remove_homodimers:
        df = df.loc[df["orf_name_f1"] != df["orf_name_f2"], :]
    return df


def load_y2h_pairwise_test(*, remove_homodimers):
    """

    Args:
        remove_homodimers (bool)

    """
    df = pd.read_csv("../data/internal/Y2H_v4_pairwise_test.tsv", sep="\t")
    df = df.drop_duplicates(subset=["orf_name_a", "orf_name_b", "source_dataset"])
    # Note: all other results are treated as missing values
    df["result"] = df["result"].map({"Positive": True, "Negative": False})
    df["assay"] = "Y2H v4"
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_additional_y2h_pairwise_test(*, remove_homodimers):
    """

    Args:
        remove_homodimers (bool)

    """
    df = pd.read_csv(
        "../data/internal/Y2H_v4_pairwise_test_AlphaFoldRoseTTAFold.tsv", sep="\t"
    )
    
    # PRS/RRS tested in both AD/DB orientations. Only keep one to compare to
    # AF/RF data which was tested in a single orientation.
    df = (df.loc[(df['same_orientation_as_previous_experiment'] == True) |
                  df['same_orientation_as_previous_experiment'].isnull(), :]
            .drop(columns=['same_orientation_as_previous_experiment']))
        
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    # Note: all other results are treated as missing values
    df["result"] = df["result"].map({"Positive": True, "Negative": False})
    df["assay"] = "Y2H v4"
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_YeRI_search_space():
    return pd.read_csv("../data/internal/ORF-search-space_YeRI.tsv", sep="\t")


def load_CCSB_YI1_search_space():
    """
    Screened space of Yu et al. Science 2008

    This data wasn't in any of the supplementary tables, so including it here.

    WARNING:
        This uses the old systematic ORF names. Need to be mapped to updated.

    """
    df = pd.read_csv("../data/internal/CCSB-YI1_space.tsv", sep="\t")
    return df


def load_cyc_2008():
    cyc = pd.read_csv("../data/external/CYC2008_complex.tab", sep="\t")
    complexes = {}
    for c, g in cyc.groupby("Complex"):
        complexes[c] = set(g["ORF"].values)
    return complexes


def load_EBI_complex_portal():
    df = pd.read_csv("../data/external/EBI_yeast_complexes.tsv", sep="\t")
    prefix_to_ignore = ("CHEBI:", "EBI-", "DIP-", "URS", "NP_")
    complexes = {}
    for i, row in df.iterrows():
        members = row["Identifiers (and stoichiometry) of molecules in complex"].split(
            "|"
        )
        members = [p for p in members if not p.startswith(prefix_to_ignore)]
        members = [p[: p.index("(")] if "(" in p else p for p in members]
        members = [p[: p.index("-PRO")] if "-PRO" in p else p for p in members]
        complexes[row["#Complex ac"]] = set(members)
    while any(v.startswith("CPX-") for vs in complexes.values() for v in vs):
        for k, vs in complexes.items():
            for v in vs.copy():
                if v.startswith("CPX-"):
                    complexes[k].update(complexes[v])
                    complexes[k].remove(v)
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map[3] = id_map[3].fillna(id_map[1])
    id_map = id_map.loc[:, [1, 3, 6]]
    id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]
    id_map = (
        id_map.drop_duplicates("uniprot_ac")
        .set_index("uniprot_ac")["orf_name"]
        .to_dict()
    )
    complexes = {
        k: {id_map[v] for v in vs if v in id_map} for k, vs in complexes.items()
    }
    complexes = {k: vs for k, vs in complexes.items() if len(vs) >= 2}
    return complexes


def load_costanzo_complexes():
    df = pd.read_excel("../data/external/Data File S12_Protein complex standard.xlsx")
    complexome = {
        row["Protein Complex Name"]: set(row["ORFs annotated to complex "].split("; "))
        for _i, row in df.iterrows()
    }

    # correcting a mistake where HSL7/YBR133C is missing with the dubious ORF YBR134W instead
    complexome["elm1-hsl1-hsl7"].remove("YBR134W")
    complexome["elm1-hsl1-hsl7"].add("YBR133C")

    return complexome


def load_Y2H_union_25(*, remove_homodimers):
    df = pd.read_csv("../data/internal/Y2H-union-25.tsv", sep="\t")

    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    if df["pair"].duplicated().any():
        raise UserWarning("Unexpected duplicates")
    df = df.set_index("pair")

    uetz_date = "2000-02-01"
    ito_date = "2001-04-10"
    yu_date = "2008-10-03"
    df["date"] = ""
    df.loc[df["CCSB-YI1"], "date"] = yu_date
    df.loc[df["Ito-core"], "date"] = ito_date
    df.loc[df["Uetz-screen"], "date"] = uetz_date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_y2h_union_08(*, remove_homodimers):
    df = load_Y2H_union_25(remove_homodimers=remove_homodimers)
    return df.loc[df["CCSB-YI1"] | df["Ito-core"] | df["Uetz-screen"], :]


def load_ito_search_space():
    """AD and DB fusions tested in Ito et al.
    Unpublished. Obtained by email.
    Returns:
        tuple(set(str), set(str)): AD and DB SGD ORF IDs
    """
    ad_path = "../data/external/Ito_AD_space_5565.txt"
    db_path = "../data/external/Ito_DB_space_5205.txt"
    with open(ad_path, "r") as f:
        ads = set(f.read().splitlines())
    with open(db_path, "r") as f:
        dbs = set(f.read().splitlines())
    return ads, dbs


def load_yi_i(*, remove_homodimers):
    df = load_Y2H_union_25(remove_homodimers=remove_homodimers)
    return df.loc[df["CCSB-YI1"], :]


def load_sys_nb(*, remove_homodimers):
    df = pd.read_csv("../data/internal/all_other_networks.tsv", sep="\t").drop(
        columns=["zone"]
    )
    df = df.loc[df["dataset"].isin(["Gavin_2002", "Krogan", "Gavin_2006"]), :]
    df = pd.concat(
        [df.loc[:, ["orf_name_a", "orf_name_b"]], pd.get_dummies(df["dataset"])], axis=1
    )
    df = df.groupby(["orf_name_a", "orf_name_b"]).any().reset_index()
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[
        df["orf_name_a"].isin(valid_orfs) & df["orf_name_b"].isin(valid_orfs), :
    ]
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_Michaelis_et_al_Nature_2023(*, remove_homodimers):
    """AP/MS dataset"""
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
    if remove_homodimers:
        apms = apms.loc[apms["orf_name_a"] != apms["orf_name_b"], :]
    return apms


def load_ho(*, remove_homodimers):
    df = pd.read_csv("../data/internal/all_other_networks.tsv", sep="\t").drop(
        columns=["zone"]
    )
    df = df.loc[df["dataset"].isin(["Ho"]), :]
    df = df.groupby(["orf_name_a", "orf_name_b"]).any().reset_index()
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[
        df["orf_name_a"].isin(valid_orfs) & df["orf_name_b"].isin(valid_orfs), :
    ]
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_tarassov(*, remove_homodimers):
    df = pd.read_csv("../data/internal/all_other_networks.tsv", sep="\t").drop(
        columns=["zone"]
    )
    df = df.loc[df["dataset"].isin(["Tarassov"]), :]
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[
        df["orf_name_a"].isin(valid_orfs) & df["orf_name_b"].isin(valid_orfs), :
    ]
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_gi_psn():
    df = pd.read_csv("../data/internal/all_other_networks.tsv", sep="\t").drop(
        columns=["zone"]
    )
    df = df.loc[df["dataset"].isin(["GI-PSN"]), :]
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[
        df["orf_name_a"].isin(valid_orfs) & df["orf_name_b"].isin(valid_orfs), :
    ]
    df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_cs_psn():
    df = pd.read_csv("../data/internal/all_other_networks.tsv", sep="\t").drop(
        columns=["zone"]
    )
    df = df.loc[df["dataset"].isin(["CS-PSN"]), :]
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[
        df["orf_name_a"].isin(valid_orfs) & df["orf_name_b"].isin(valid_orfs), :
    ]
    df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_ge_psn():
    df = pd.read_csv("../data/internal/all_other_networks.tsv", sep="\t").drop(
        columns=["zone"]
    )
    df = df.loc[df["dataset"].isin(["GE-PSN"]), :]
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[
        df["orf_name_a"].isin(valid_orfs) & df["orf_name_b"].isin(valid_orfs), :
    ]
    df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_sga_pcc_matrix():
    """PCC matrix from double-deletion in yeast data from the Boone lab.
    See [1]_
    References:
        .. [1] Costanzo et al. (2016). A global genetic interaction network
           maps a wiring diagram of cellular function, Science 353(6306).
    Returns:
        pandas.DataFrame: PCC matrix where columns and index are SGD ORF IDs
    See Also:
        :func:`load_nw_sga` convert to network by applying a PCC cutoff
    """
    fpath = "../data/external/cc_ALL.txt"
    sga = pd.read_csv(fpath, index_col=1, skiprows=1, sep="\t")
    sga = sga.drop(columns=sga.columns[0])  # gene names column
    # duplicate columns in file, which pandas doesn't support in reading to
    # DataFrame, it automatically appends .1, .2, ... so have to undo that
    sga.columns = [c.split(".")[0] for c in sga.columns]
    return sga


def correlation_matrix_to_network(corrs, cutoff, id_type):
    """Covert correlations to table of interactions by applying a threshold.
    Args:
        corrs (pandas.DataFrame): correlation matrix
        cutoff (float): minimum threshold to define an interaction
        id_type (str): gene/protein identifier type used
    Returns:
        pandas.DataFrame: table of interactions, one row per interaction
    """
    if cutoff > 1.0 or cutoff < -1.0:
        raise ValueError("Invalid value for PCC cutoff: " + str(cutoff))
    nw = corrs[corrs > cutoff].stack().reset_index()
    nw.columns = [id_type + "_a", id_type + "_b", "PCC"]
    a = nw[[id_type + "_a", id_type + "_b"]].min(axis=1)
    b = nw[[id_type + "_a", id_type + "_b"]].max(axis=1)
    nw[id_type + "_a"] = a
    nw[id_type + "_b"] = b
    nw = nw.drop_duplicates()
    if nw[[id_type + "_a", id_type + "_b"]].duplicated().sum() > 0:
        # Take higher PCC in case where duplicate measurements exist
        nw = nw.groupby([id_type + "_a", id_type + "_b"]).max().reset_index()
    nw = nw.loc[nw[id_type + "_a"] != nw[id_type + "_b"], :]
    nw = nw.set_index(
        nw[id_type + "_a"].astype(str) + "_" + nw[id_type + "_b"].astype(str)
    )
    return nw


def correlation_matrix_to_pairs_table(corrs, id_type):
    """Covert correlations to table of interactions by applying a threshold.
    Args:
        corrs (pandas.DataFrame): correlation matrix
        cutoff (float): minimum threshold to define an interaction
        id_type (str): gene/protein identifier type used
    Returns:
        pandas.DataFrame: table of interactions, one row per interaction
    """
    nw = (
        corrs.stack()
        .rename_axis([id_type + "_a", id_type + "_b"])
        .rename("PCC")
        .reset_index()
    )
    a = nw[[id_type + "_a", id_type + "_b"]].min(axis=1)
    b = nw[[id_type + "_a", id_type + "_b"]].max(axis=1)
    nw[id_type + "_a"] = a
    nw[id_type + "_b"] = b
    nw = nw.drop_duplicates()
    if nw[[id_type + "_a", id_type + "_b"]].duplicated().sum() > 0:
        # Take mean PCC in case where duplicate measurements exist
        nw = nw.groupby([id_type + "_a", id_type + "_b"]).mean().reset_index()
    nw = nw.loc[nw[id_type + "_a"] != nw[id_type + "_b"], :]
    nw = nw.set_index(
        nw[id_type + "_a"].astype(str) + "_" + nw[id_type + "_b"].astype(str)
    )
    return nw


def load_old_systematic_orf_name_updates():
    name_map = pd.read_csv("../data/external/SGD_ORFs_2017-01-14.tsv", sep="\t")
    name_map["alias"] = name_map["alias"].str.split("|", expand=False)
    name_map = name_map.explode("alias")
    name_map = name_map.dropna(subset=["alias"])
    # Examples of old systematic ORF names that dont match current rules:
    # YCLX08C YOR29-06 YCRX13W YGL023 YHR039BC YHR079BC YOR3165W
    old_nuclear_regexp = "Y[A-P][LR](X)?[0-9]{1,5}([CW])?(-[A-Z0-9]{1,4})?"
    mito_regexp = "Q[0-9]{4}"
    two_micron_regexp = "R[0-9]{4}[CW]"
    OLD_SGD_ORF_NAME_REGEXP = "({})|({})|({})".format(
        old_nuclear_regexp, mito_regexp, two_micron_regexp
    )
    name_map = name_map.loc[name_map["alias"].str.match(OLD_SGD_ORF_NAME_REGEXP), :]
    if name_map["alias"].duplicated().any():
        raise UserWarning("Unexpected duplicates")
    name_map = name_map.set_index("alias")["orf_name"].to_dict()
    # these 3 pairs of ORFs were accidently switched at some point in SGD
    switched_orfs = [
        "YLR154W-C",
        "YLR154W-A",
        "YNL067W-B",
        "YNL067W-A",
        "YOL013W-A",
        "YOL013W-B",
    ]
    for switched_orf in switched_orfs:
        name_map[switched_orf] = switched_orf
    return name_map


def update_old_systematic_yeast_orf_names_network(nw):
    nw_new = nw.copy()
    updated_orf_name_map = load_old_systematic_orf_name_updates()
    nw_new["orf_name_a"] = nw_new["orf_name_a"].map(
        lambda x: updated_orf_name_map.get(x, x)
    )
    nw_new["orf_name_b"] = nw_new["orf_name_b"].map(
        lambda x: updated_orf_name_map.get(x, x)
    )
    nw_new.index = nw_new["orf_name_a"] + "_" + nw_new["orf_name_b"]
    if nw_new.index.duplicated().any():
        raise UserWarning("Unexpected duplicates")
    return nw_new


def update_old_systematic_yeast_orf_names_matrix(nw):
    nw_new = nw.copy()
    updated_orf_name_map = load_old_systematic_orf_name_updates()
    nw_new.index = nw_new.index.map(lambda x: updated_orf_name_map.get(x, x))
    nw_new.columns = nw_new.columns.map(lambda x: updated_orf_name_map.get(x, x))
    return nw_new


def restrict_to_SGD_verified_and_uncharacterized_protein_coding_ORFs(nw):
    nw_new = nw.copy()
    sgd_space = load_non_dubious_orfs()
    nw_new = nw_new.loc[
        nw_new["orf_name_a"].isin(sgd_space) & nw_new["orf_name_b"].isin(sgd_space), :
    ]
    return nw_new


def load_gi_pcc_values():
    corrs = load_sga_pcc_matrix()
    corrs = update_old_systematic_yeast_orf_names_matrix(corrs)
    df = correlation_matrix_to_pairs_table(corrs, "orf_name")
    if df.index.duplicated().any():
        raise UserWarning("Unexpected duplicates")
    df = restrict_to_SGD_verified_and_uncharacterized_protein_coding_ORFs(df)
    if df.index.duplicated().any():
        raise UserWarning("Unexpected duplicates")
    return df


def load_cs_pcc_values():
    df = pd.read_csv("../data/external/hom.ratio_result_nm.pub", sep="\t")
    df["Orf"] = df["Orf"].str.replace(":chr.*", "", regex=True)
    df = df.set_index("Orf")
    # NOTE: here we drop all rows with missing values.
    # which keeps 4014 of 4753 (84%) unique ORFs
    # Could have instead required some minimum number of data points
    df = df.dropna()
    corrs = df.T.corr()
    corrs = update_old_systematic_yeast_orf_names_matrix(corrs)
    df = correlation_matrix_to_pairs_table(corrs, "orf_name")
    df = restrict_to_SGD_verified_and_uncharacterized_protein_coding_ORFs(df)
    if df.index.duplicated().any():
        raise UserWarning("Unexpected duplicates")
    return df


def load_ge_pcc_values():
    ge = pd.read_csv('../data/processed/GE_PCC_values.tsv', 
                     sep='\t',
                     index_col=0)
    return ge



def load_gi_search_space():
    pcc = load_gi_pcc_values()
    return set(pcc["orf_name_a"].unique()).union(pcc["orf_name_b"].unique())


def load_cs_search_space():
    pcc = load_cs_pcc_values()
    return set(pcc["orf_name_a"].unique()).union(pcc["orf_name_b"].unique())


def load_ge_search_space():
    pcc = load_ge_pcc_values()
    return set(pcc["orf_name_a"].unique()).union(pcc["orf_name_b"].unique())


def load_lit_13(*, remove_homodimers):
    df = pd.read_csv("../data/internal/Lit-13-yeast.tsv", sep="\t")
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_lit_17(*, remove_homodimers):
    df = pd.read_csv("../data/internal/Lit-17-yeast.tsv", sep="\t")
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    return df


def load_lit_20(*, remove_homodimers, include_date=False):
    """

    This is the literature curated PPI data, with proteome-wide systematic maps removed.
    Those maps are: Ito-core, Uetz-library, YI-I (Yu et al.), Tarassov et al. Gavin et al. 2002
    and 2006, Ho and Krogan.

    """
    df = pd.read_csv("../data/internal/Lit-20-yeast.tsv", sep="\t")
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    if include_date:
        df = _add_date_to_lit(df)
    return df


def load_lit_24(*, remove_homodimers, include_date=False):
    """

    This is the literature curated PPI data, with proteome-wide systematic maps removed.
    Those maps are: Ito-core, Uetz-library, YI-I (Yu et al.), Tarassov et al. Gavin et al. 2002
    and 2006, Ho and Krogan.

    """
    df = pd.read_csv("../data/internal/Lit-24-yeast.tsv", sep="\t")
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.set_index("pair")
    if remove_homodimers:
        df = df.loc[df["orf_name_a"] != df["orf_name_b"], :]
    if include_date:
        df = _add_date_to_lit(df)
    return df


def load_lit_bm_13(*, remove_homodimers):
    df = load_lit_13(remove_homodimers=remove_homodimers)
    df = df.loc[df["category"] == "Lit-BM", :]
    df = df.drop(columns=["category"])
    return df


def load_lit_bm_17(*, remove_homodimers):
    df = load_lit_17(remove_homodimers=remove_homodimers)
    df = df.loc[df["category"] == "Lit-BM", :]
    df = df.drop(columns=["category"])
    return df


def load_lit_bm_20(*, remove_homodimers, include_date=False):
    df = load_lit_20(remove_homodimers=remove_homodimers, include_date=include_date)
    df = df.loc[df["category"] == "Lit-BM", :]
    df = df.drop(columns=["category"])
    return df


def load_lit_bm_24(*, remove_homodimers, include_date=False):
    df = load_lit_24(remove_homodimers=remove_homodimers, include_date=include_date)
    df = df.loc[df["category"] == "Lit-BM", :]
    df = df.drop(columns=["category"])
    return df


def _add_date_to_lit(lit):
    evid = pd.read_csv("../data/internal/Lit-24-yeast_evidence.tsv", sep="\t")
    pmid_date = load_pubmed_to_date_mapping('../data/processed/pmid_dates_yeast.csv')
    if evid["pubmed_id"].max() > pmid_date.index.max():
        warnings.warn("Pubmed ID to date mapping is too old")
    evid["pair"] = evid["orf_name_a"] + "_" + evid["orf_name_b"]
    evid["date"] = evid["pubmed_id"].map(pmid_date)
    first_date = (
        evid.sort_values("date").drop_duplicates("pair").set_index("pair")["date"]
    )
    lit["date_first_pub"] = lit.index.map(first_date)
    second_date = evid.sort_values("date").groupby("pair").nth(1)["date"]
    lit["date_second_pub"] = lit.index.map(second_date)
    first_binary_date = (
        evid.loc[(evid["binary"] == 1), :].sort_values("date").drop_duplicates("pair").set_index("pair")["date"]
    )
    lit["date_first_binary_pub"] = lit.index.map(first_binary_date)
    lit["date_bm"] = lit[["date_first_binary_pub", "date_second_pub"]].max(axis=1)

    lit = lit.sort_values("date_first_pub")
    return lit


def load_lit_20_evidence():
    df = pd.read_csv("../data/internal/Lit-20-yeast_evidence.tsv", sep="\t")
    return df


def load_lit_24_evidence():
    df = pd.read_csv("../data/internal/Lit-24-yeast_evidence.tsv", sep="\t")
    return df


def load_pdb_id_to_date(pdb_ids=None):
    """

    NOTE: the API converts the PDB ID to lower case
    even if you query an upper case ID

    """
    if pdb_ids is None:
        pdb_ids = set()
    pdb_ids = set([x.lower() for x in pdb_ids])
    cache_file = Path("../data/external/pdb_id_release_date.tsv")
    if cache_file.exists():
        with open(cache_file, "r") as f:
            pdb_to_date = dict(line.strip().split("\t") for line in f)
        if set(pdb_ids).issubset(set(pdb_to_date.keys())):
            return pdb_to_date
    else:
        pdb_to_date = {}
    print("Calling PDB API for release dates -- will take some time")
    pdbe = bioservices.PDBe()
    for pdb_id in tqdm.tqdm(pdb_ids):
        if pdb_id in pdb_to_date:
            continue
        res = pdbe.get_summary(pdb_id)
        if pdb_id in res:
            pdb_to_date[pdb_id] = res[pdb_id][0]["release_date"]
        else:
            pdb_to_date[pdb_id] = np.nan

    with open(cache_file, "w") as f:
        for k, v in pdb_to_date.items():
            f.write(k + "\t" + str(v) + "\n")

    return pdb_to_date


def load_I3D_exp_24(*, remove_homodimers, include_date=False):
    """
    Patrick's lab re-ran the I3D pipeline for this paper.
    """
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map[3] = id_map[3].fillna(id_map[1])
    id_map = id_map.loc[:, [1, 3, 6]]
    id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]
    # map uniprot to SGD, get pairs
    i3d = pd.read_csv(
        "../data/internal/interactome3d_2024_Nov_scer_interactions_complete.dat",
        sep="\t",
    )

    if include_date:
        pdb_to_date = load_pdb_id_to_date(set(i3d["PDB_ID"].unique()))
        i3d["date"] = pd.to_datetime(i3d["PDB_ID"].map(pdb_to_date), errors="coerce")
        # so that when we drop duplicates below, we keep earliest date
        i3d = i3d.sort_values("date")

    i3d = i3d.loc[i3d["TYPE"] == "Structure", :]
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
    valid_orfs = load_non_dubious_orfs()
    i3d = i3d.loc[
        i3d["orf_name_a"].isin(valid_orfs) & i3d["orf_name_b"].isin(valid_orfs), :
    ]
    if remove_homodimers:
        i3d = i3d.loc[i3d["orf_name_a"] != i3d["orf_name_b"], :]
    return i3d


def load_I3D_exp_20(*, remove_homodimers, include_date=False):
    """

    Current release as of 2023-08-08. 2020-05 refers to the UniProt vesion when
    the release was generated, at the end of 2020.
    Only experimental structures. No models.

    """
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map[3] = id_map[3].fillna(id_map[1])
    id_map = id_map.loc[:, [1, 3, 6]]
    id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]
    # map uniprot to SGD, get pairs
    i3d = pd.read_csv(
        "../data/external/interactome3d_2020-05_scer_interactions.dat", sep="\t"
    )

    if include_date:
        df = pd.read_csv(
            "../data/external/interactome3d_2020-05_scer_interactions_complete.dat",
            sep="\t",
        )
        pdb_to_date = load_pdb_id_to_date()
        df["date"] = pd.to_datetime(df["PDB_ID"].map(pdb_to_date), errors="coerce")
        i3d = pd.merge(
            i3d,
            df.groupby(["PROT1", "PROT2"])["date"].min().reset_index(),
            how="left",
            on=["PROT1", "PROT2"],
        )

    i3d = i3d.loc[i3d["TYPE"] == "Structure", :]
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
    valid_orfs = load_non_dubious_orfs()
    i3d = i3d.loc[
        i3d["orf_name_a"].isin(valid_orfs) & i3d["orf_name_b"].isin(valid_orfs), :
    ]
    if remove_homodimers:
        i3d = i3d.loc[i3d["orf_name_a"] != i3d["orf_name_b"], :]
    return i3d


def load_i3d_exp_17(*, remove_homodimers):
    """

    Release used for Y2H experiment from June 2017. Only experimental structures. No models.

    """
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map[3] = id_map[3].fillna(id_map[1])
    id_map = id_map.loc[:, [1, 3, 6]]
    id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]
    # map uniprot to SGD, get pairs
    i3d = pd.read_csv(
        "../data/external/interactome3d_2017_June_scer_interactions.dat", sep="\t"
    )
    i3d = i3d.loc[i3d["TYPE"] == "Structure", :]
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
    valid_orfs = load_non_dubious_orfs()
    i3d = i3d.loc[
        i3d["orf_name_a"].isin(valid_orfs) & i3d["orf_name_b"].isin(valid_orfs), :
    ]
    if remove_homodimers:
        i3d = i3d.loc[i3d["orf_name_a"] != i3d["orf_name_b"], :]
    return i3d


def load_full_interactome_3d_2020(*, remove_homodimers):
    """
    Includes homology models.
    """
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
    if remove_homodimers:
        i3d = i3d.loc[i3d["orf_name_a"] != i3d["orf_name_b"], :]
    return i3d


def load_direct_and_indirect_pairs():
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map[3] = id_map[3].fillna(id_map[1])
    id_map = id_map.loc[:, [1, 3, 6]]
    id_map.columns = ["orf_name", "gene_name", "uniprot_ac"]
    # map uniprot to SGD, get pairs
    indirect = pd.read_csv(
        "../data/internal/direct_vs_indirect_contacts_complex_size_gte_3.tsv", sep="\t"
    )
    indirect = pd.merge(
        indirect, id_map, how="inner", left_on="uniprot_ac_a", right_on="uniprot_ac"
    )
    indirect = pd.merge(
        indirect,
        id_map,
        how="inner",
        left_on="uniprot_ac_b",
        right_on="uniprot_ac",
        suffixes=("_1", "_2"),
    )
    indirect["pair"] = (
        indirect[["orf_name_1", "orf_name_2"]].min(axis=1)
        + "_"
        + indirect[["orf_name_1", "orf_name_2"]].max(axis=1)
    )
    if indirect["pair"].duplicated().any():
        raise UserWarning("Unexpected duplicate pairs")
    indirect = indirect.set_index("pair").drop(columns=["uniprot_ac_a", "uniprot_ac_b"])
    return indirect


def load_interface_area():
    df = pd.read_csv("../data/internal/interface_surface_area.tsv", sep="\t")
    id_map = pd.read_csv(
        "../data/external/sgd_uniprot_mapping.tsv", sep="\t", header=None
    )
    id_map = id_map.loc[:, [1, 6]]
    id_map.columns = ["orf_name", "uniprot_ac"]
    df = pd.merge(
        df, id_map, how="inner", left_on="uniprot_ac_a", right_on="uniprot_ac"
    )
    df = pd.merge(
        df,
        id_map,
        how="inner",
        left_on="uniprot_ac_b",
        right_on="uniprot_ac",
        suffixes=("_1", "_2"),
    )
    df["orf_name_a"] = df[["orf_name_1", "orf_name_2"]].min(axis=1)
    df["orf_name_b"] = df[["orf_name_1", "orf_name_2"]].max(axis=1)
    df["pair"] = df["orf_name_a"] + "_" + df["orf_name_b"]
    df = df.drop_duplicates("pair")
    return df.set_index("pair")["interface_area"]


def load_3did_DDIs():
    fpath = '../data/external/3did_flat_2024-12'
    domain_pairs = []
    for line in open(fpath, 'r'):
        if line.startswith('#=ID'):
            pfam_a, pfam_b = ((line.split()[3][1:8], line.split()[4][:7]))
        if line.startswith('#=3D'):
            domain_pairs.append((line.split()[1], 
                                 line.split()[2].split(':')[0], 
                                 line.split()[3].split(':')[0],
                                 pfam_a,
                                 pfam_b))
    df = pd.DataFrame(data=domain_pairs, columns=['pdb_id', 'chain_1', 'chain_2', 'pfam_1', 'pfam_2'])
    df = df.drop_duplicates()
    if df.duplicated().any():
        raise UserWarning('unexpected duplicates')
    return df


def load_protein_properties():
    df = pd.read_csv("../data/internal/protein_properties.tsv", sep="\t")
    valid_orfs = load_non_dubious_orfs()
    df = df.loc[df["orf_name"].isin(valid_orfs), :]
    if df["orf_name"].duplicated().any():
        raise UserWarning("Duplicate gene names")
    df = df.set_index("orf_name")
    return df


def load_essential_genes():
    df = load_protein_properties()
    return set(df.loc[df["Essential"] == 1, :].index)


def load_genetic_interactions():
    # return list of positive and negative pairs
    # should be 550k negative and 350k positive
    neg_gi = set([])
    pos_gi = set([])
    for file_name in ["SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]:
        df = pd.read_csv("../data/external/" + file_name, sep="\t")
        df["systematic_name_query"] = df["Query Strain ID"].str.split(
            "_", n=1, expand=True
        )[0]
        df["systematic_name_array"] = df["Array Strain ID"].str.split(
            "_", n=1, expand=True
        )[0]
        df["pair"] = (
            df[["systematic_name_query", "systematic_name_array"]].min(axis=1)
            + "_"
            + df[["systematic_name_query", "systematic_name_array"]].max(axis=1)
        )
        pos_gi = pos_gi.union(
            set(
                df.loc[
                    (df["Genetic interaction score (ε)"] > 0.08)
                    & (df["P-value"] < 0.05),
                    "pair",
                ].values
            )
        )
        neg_gi = neg_gi.union(
            set(
                df.loc[
                    (df["Genetic interaction score (ε)"] < -0.08)
                    & (df["P-value"] < 0.05),
                    "pair",
                ].values
            )
        )
    return neg_gi, pos_gi


def load_ribosome_related_proteins():
    df = pd.read_excel(
        "../data/external/Benschop_Mol_Cell_2010_table_S1.xls",
        sheet_name="Curated consensus + GO",
    )
    ribosome_related_complex_names = [
        "mitochondtrial large ribosomal subunit",
        "mitochondrial small ribosomal subunit",
        "cytosolic small ribosomal subunit",
        "cytosolic large ribosomal subunit",
        "90S preribosome",
        "preribosome, large subunit precursor",
        "90S preribosome (GO)",
        "mitochondrial ribosome (GO)",
        "nucleolar preribosome, large subunit precursor (GO)",
        "nucleolar preribosome, small subunit precursor (GO)",
        "preribosome (GO)",
    ]
    df = df.loc[
        df["# Complex name"].isin(ribosome_related_complex_names),
        ["# Complex name", "Complex members (systematic name)"],
    ]

    ribo = {
        row["# Complex name"]: set(row["Complex members (systematic name)"].split("; "))
        for i, row in df.iterrows()
    }
    orfs = load_all_orfs()
    ribo["cytosolic large ribosomal subunit"].update(
        set(
            orfs.loc[
                orfs["gene_name"].str.startswith("RPL").fillna(False), "orf_name"
            ].values
        )
    )
    ribo["cytosolic small ribosomal subunit"].update(
        set(
            orfs.loc[
                orfs["gene_name"].str.startswith("RPS").fillna(False), "orf_name"
            ].values
        )
    )
    # all_ribo.update(set(orfs.loc[(orfs['gene_name'].str.startswith('RPL') |
    #                orfs['gene_name'].str.startswith('RPS')), 'orf_name'].values))
    # two manual edits: NSA3-->CIC1, PWP1A-->PWP1
    mccann_genes_dev_2015_table_1 = """
            Rix7
            Rrp43
            Arx1, Rrp12
            Nog1, Nog2, Nug1
            Dbp10, Dbp3, Dbp6, Dbp7, Dbp9,
            Drs1, Has1, Mak5, Prp43, Spb4 Grc3
            Nop2, Spb1
            Fpr3, Fpr4
            Brx1, Bud20, Cgr1, Ebp2, Erb1, Las1,
            Loc1, Mak11, Mak16, Mak21, Mrt4, Nip7, Noc2, Noc3, Nop12, Nop13, Nop15, Nop16, Nop4, Nop53, Nop7, Nop8, Nsa1, Nsa2, CIC1, Nsr1, Puf6, Pwp1a, Rlp24, Rlp7, Rpf1, Rpf2, Rrb1, Rrp1, Rrp14, Rrp15, Rrp5, Rrp8, Rrs1, Rsa3, Rsa4, Ssf1, Ssf2, Tif6, Tma16, Urb1, Urb2, Ytm1
            """
    lsu_processome = mccann_genes_dev_2015_table_1.upper().replace(",", "").split()
    lsu_processome = set(
        orfs.loc[orfs["gene_name"].isin(lsu_processome), "orf_name"].values
    )
    ribo["LSU processome"] = lsu_processome
    ribo_merged = {
        "mitochondrial ribosome": [
            "mitochondtrial large ribosomal subunit",
            "mitochondrial small ribosomal subunit",
        ],
        "cytosolic ribosome": [
            "cytosolic small ribosomal subunit",
            "cytosolic large ribosomal subunit",
        ],
        "preribosome": [
            "90S preribosome",
            "preribosome, large subunit precursor",
            "nucleolar preribosome, large subunit precursor (GO)",
            "nucleolar preribosome, small subunit precursor (GO)",
            "preribosome (GO)",
            "LSU processome",
        ],
    }
    ribo_merged = {k: set.union(*[ribo[v] for v in l]) for k, l in ribo_merged.items()}
    ribo_merged["preribosome"] = ribo_merged["preribosome"].difference(
        ribo_merged["cytosolic ribosome"]
    )
    # all_ribo = set.union(*list(ribo.values()))
    return ribo_merged


def load_subcellular_location():
    """
    - CYCLoPS data - three WT repeats
    - take all non-zero values as observation of being in that
    compartment
    - take observation in any of the repeats

    """
    repeats = []
    for i in range(1, 4):
        df = pd.read_excel(
            "../data/external/CYCLoPs Table3-{i}_WT{i}_LOCscore.xls".format(i=i),
            skiprows=3,
        )
        if df["ORF"].duplicated().any():
            raise UserWarning("Duplicate entries for ORF")
        df = df.set_index("ORF").drop(columns=["Gene name"])
        df = df > 0.0
        locs = df.columns
        df = df.rename(columns={c: c + "_" + str(i) for c in df.columns})
        repeats.append(df)
    df = repeats[0].join(repeats[1:], how="outer", sort=True)
    for comp in locs:
        df[comp] = df[comp + "_1"] | df[comp + "_2"] | df[comp + "_3"]
    df = df.loc[:, locs]
    if df.isnull().any().any():
        raise UserWarning("Something went wrong")
    return df


def load_number_of_publications_per_gene():
    df = pd.read_csv("../data/internal/protein_properties.tsv", sep="\t")
    return df.set_index("orf_name")["Publication count"]


def load_unstudied_genes():
    """

    Table S9 from
    Wood et al. Hidden in Plain Sight: What Remains to Be Discovered in the Eukaryotic Proteome?, Open Biology, 2019

    """
    return set(
        pd.read_csv("../data/external/rsob180241_si_010.tsv")[
            "Identifier (UniProtKB Accession or ORF Name)"
        ].values
    )


def load_kegg_pathway_genes():
    s = KEGG()
    yeast_pathways = [l.split()[0] for l in s.list('pathway', organism='sce').splitlines()]
    pathway_genes = {}
    for pathway_id in yeast_pathways:
        data = s.get(pathway_id)
        pathway_genes[pathway_id] = set()
        for line in data.splitlines():
            section = line[:12].strip() if line[:12].strip() != '' else section
            if section == 'GENE':
                pathway_genes[pathway_id].add(line[12:].split()[0])
    return pathway_genes


def load_AlphaFold_RoseTTAFold(
    remove_homodimers=True, restrict_to_high_confidence=True
):
    """Yeast PPIs with predicted structures

    Humphreys et al. Science Nov 2021

    """
    if not remove_homodimers:
        raise ValueError("There are no self interactions in this dataset")
    afrf = pd.read_excel(
        "../data/external/science.abm4805_predicted_protein_protein_interactions.xlsx"
    )
    orfs = load_all_orfs()
    orfs["gene_name"] = orfs["gene_name"].fillna(orfs["orf_name"])
    gene_to_orf_name = orfs.dropna(subset=["gene_name"]).set_index("gene_name")[
        "orf_name"
    ]
    afrf["orf_name1"] = (
        afrf["gene name1"].map(gene_to_orf_name).fillna(afrf["gene name1"])
    )
    afrf["orf_name2"] = (
        afrf["gene name2"].map(gene_to_orf_name).fillna(afrf["gene name2"])
    )
    missing_names = (afrf["orf_name1"] == "na") | (afrf["orf_name2"] == "na")
    if missing_names.sum() > 0:
        print(
            "NOTE: dropping {} pairs with missing ORF names".format(
                missing_names.sum()
            )
        )
    afrf = afrf.loc[~missing_names, :]
    if afrf.loc[:, ["orf_name1", "orf_name2"]].isnull().any().any():
        raise UserWarning("Failure to map gene names")
    afrf["orf_name_a"] = afrf.loc[:, ["orf_name1", "orf_name2"]].min(axis=1)
    afrf["orf_name_b"] = afrf.loc[:, ["orf_name1", "orf_name2"]].max(axis=1)
    afrf["pair"] = afrf["orf_name_a"] + "_" + afrf["orf_name_b"]
    if afrf["pair"].duplicated().any():
        raise UserWarning("unexpected duplicates")
    afrf = afrf.set_index("pair")
    if restrict_to_high_confidence:
        afrf = afrf.loc[afrf["PPI Score"] > 0.95, :]
    return afrf


def load_afrf_yeast_search_space():
    df = pd.read_csv('../data/personal_communication/screen_list',
                    header=None,
                    names=['orthodb_pair'])
    df[['orthodb_id_x', 'orthodb_id_y']] = df['orthodb_pair'].str.split('_', expand=True)
    id_map = pd.read_csv('../data/internal/orthoDB_ID_to_yeast_ORF_name.txt',
                        sep='\t')
    if id_map['OrthoDB_ID'].duplicated().any():
        raise UserWarning('Duplicated OrthoDB_IDs in mapping file')
    id_map = id_map.set_index('OrthoDB_ID')['Yeast_ORF'].to_dict()
    df['orf_name_x'] = df['orthodb_id_x'].map(id_map)
    df['orf_name_y'] = df['orthodb_id_y'].map(id_map)
    if df.isnull().any().any():
        raise UserWarning('Some OrthoDB IDs could not be mapped to yeast ORF names')
    df['orf_name_a'] = df[['orf_name_x', 'orf_name_y']].min(axis=1)
    df['orf_name_b'] = df[['orf_name_x', 'orf_name_y']].max(axis=1)
    df['pair'] = df['orf_name_a'] + '_' + df['orf_name_b']
    df = df.drop(columns=['orthodb_pair', 'orthodb_id_x', 'orthodb_id_y', 'orf_name_x', 'orf_name_y'])
    if df.duplicated(subset=['pair']).any():
        raise UserWarning('Unexpected duplicate pairs')
    df = df.set_index('pair')
    valid_orfs = load_non_dubious_orfs()
    if not (df['orf_name_a'].isin(valid_orfs) & df['orf_name_b'].isin(valid_orfs)).all():
        raise UserWarning('Some ORF names are not valid ORFs')
    return df


def load_PDB_metadata(pdb_ids):

    """date, method, size of structure etc."""
    pdb_ids = set([x.lower() for x in pdb_ids])
    f_path = Path("../data/external/pdb_metadata.tsv")
    if f_path.exists():
        df = pd.read_csv(f_path, sep="\t", index_col=0)
        if set(pdb_ids).issubset(set(df.index)):
            return df
        else:
            data = df.T.to_dict()
    else:
        data = {}
    pdbe = bioservices.PDBe()
    for pdb_id in pdb_ids:
        # NOTE there are a small number of PDB IDs that have been superceeded
        # since the Interactome3D file was generated, e.g. 3J3B
        if pdb_id in data:
            continue
        else:
            data[pdb_id] = {}
        pdbe_summary = pdbe.get_summary(pdb_id).get(pdb_id, [None])[0]
        pdbe_assemblies = pdbe.get_assembly(pdb_id)
        if pdbe_summary is None:
            for c in [
                "date",
                "n_polypeptide",
                "n_dna",
                "n_rna",
                "n_sugar",
                "n_ligand",
                "n_carbohydrate_polymer",
                "experimental_method",
                "polymeric_count",
            ]:
                data[pdb_id][c] = np.nan
            continue
        data[pdb_id]["date"] = pdbe_summary["release_date"]
        for x in [
            "polypeptide",
            "dna",
            "rna",
            "sugar",
            "ligand",
            "carbohydrate_polymer",
        ]:
            data[pdb_id][f"n_{x}"] = pdbe_summary["number_of_entities"][x]
        data[pdb_id]["experimental_method"] = pdbe_summary["experimental_method"][0]
        for i, assembly in enumerate(pdbe_summary["assemblies"]):
            if assembly["preferred"]:
                data[pdb_id]["polymeric_count"] = pdbe_assemblies[pdb_id][i][
                    "polymeric_count"
                ]
        if "polymeric_count" not in data[pdb_id]:
            raise UserWarning(f"No preferred assembly for {pdb_id}")
        continue
    df = pd.DataFrame(data).T
    df.to_csv(f_path, index=True, sep="\t")
    return df


def load_double_edge_YeRI(*, remove_homodimers):
    abbi = load_Y2H_union_25(remove_homodimers=remove_homodimers)
    val = load_GPCA_and_MAPPIT_data(remove_homodimers=remove_homodimers)
    abbi["n_datasets"] = (
        abbi["Uetz-screen"].astype(int)
        + abbi["Ito-core"].astype(int)
        + abbi["CCSB-YI1"].astype(int)
        + abbi["YeRI"].astype(int)
    )
    abbi["YeRI_and_MAPPIT_and_GPCA"] = abbi.index.isin(
        val.loc[
            (val["source_dataset"] == "YeRI")
            & (val["result_at_0_RRS"] == True)
            & (val["assay"] == "GPCA"),
            "pair",
        ].unique()
    ) & abbi.index.isin(
        val.loc[
            (val["source_dataset"] == "YeRI")
            & (val["result_at_0_RRS"] == True)
            & (val["assay"] == "MAPPIT"),
            "pair",
        ].unique()
    )
    abbi["YeRI_and_MAPPIT_or_GPCA"] = abbi.index.isin(
        val.loc[
            (val["source_dataset"] == "YeRI") & (val["result_at_0_RRS"] == True), "pair"
        ].unique()
    )
    dbl = abbi.loc[abbi["YeRI_and_MAPPIT_or_GPCA"], :].copy()
    return dbl


# HACK copy-pasted this from utils.py here to avoid circular import
def degree_per_protein(df, id_a="orf_id_a", id_b="orf_id_b"):
    """Calculate degree of each ORF
    Counts once for a self-interaction.
    Args:
        df (DataFrame): PPI data, one row per pair.
    Returns:
        Series: degree per protein
    """
    orfs = np.unique(np.concatenate([df[id_a].unique(), df[id_b].unique()]))
    d = pd.Series(index=orfs, data=np.zeros(orfs.shape))
    a = df.groupby(id_a).size()
    b = df.groupby(id_b).size()
    homo = df.loc[(df[id_a] == df[id_b]), id_a].values
    d.loc[a.index] = a
    d.loc[b.index] = d.loc[b.index] + b
    d.loc[homo] = d.loc[homo] - 1
    d.name = "degree"
    return d


def load_zhang_et_al_biorxiv():
    """
    
    """
    cache_file = Path('../data/processed/Zhang-et-al_2024_with-strategies.tsv')
    if cache_file.exists():
        return pd.read_csv(cache_file, sep='\t')
    df = pd.read_csv('../data/external/Zhang-et-al_biorxiv-2024_pairs.txt',
                      sep='\t', skiprows=4)
    ppi = pd.read_csv('../data/personal_communication/ppiDB_pairs.txt',
                        names=['uniprot_ac_a', 'uniprot_ac_b', '???', '????'],
                        sep='\t')
    gi = pd.read_csv('../data/personal_communication/string_pairs.txt',
                        names=['uniprot_ac_a', 'uniprot_ac_b', 'source', '????'],
                        sep='\t')
    dca = pd.read_csv('../data/external/Zhang-et-al_biorxiv-2024_DCA-scores.txt',
                    sep='\t',
                    names=['pair_id', 'DCA_score'])
    if not (df['#Protein1'] <= df['Protein2']).all():
        raise UserWarning('expected columns to be sorted')
    if not (ppi['uniprot_ac_a'] <= ppi['uniprot_ac_b']).all():
        raise UserWarning('expected columns to be sorted')
    if not (gi['uniprot_ac_a'] <= gi['uniprot_ac_b']).all():
        raise UserWarning('expected columns to be sorted')
    if dca['pair_id'].duplicated().any():
        raise UserWarning('unexpected duplicates')
    df['pair_id'] = df['#Protein1'] + '_' + df['Protein2']
    ppi['pair_id'] = ppi['uniprot_ac_a'] + '_' + ppi['uniprot_ac_b']
    gi['pair_id'] = gi['uniprot_ac_a'] + '_' + gi['uniprot_ac_b']
    df['in_PPIDB'] = df['pair_id'].isin(set(ppi['pair_id'].values))
    df['in_GI'] = df['pair_id'].isin(set(gi['pair_id'].values))
    dca = dca.set_index('pair_id')['DCA_score']
    df['DCA_score'] = df['pair_id'].map(dca)

    # There are 25 pairs missing a DCA score
    # from the allDBs column, 23 of them showed up in
    # the screens, so must be DCA > 0.25, for the other
    # two, I don't know
    df['DCA_score'] = df['DCA_score'].fillna(999)

    def shared_location(row):
        locs_a = set(row['Locality1'].split(','))
        locs_b = set(row['Locality2'].split(','))
        if locs_a == set(['none']) or locs_b == set(['none']):
            return 'CCunk'
        elif len(locs_a.intersection(locs_b)) > 0:
            return 'CCcom'
        else:
            return 'different'
        
    df['shared_location'] = df.apply(shared_location, axis=1)
    df['strategy_1'] = (
        (df['shared_location'] == 'CCcom') 
        & (df['DCA_score'] >= 0.12)
        & (df['RFprob'] >= 0.994)
    )

    df['strategy_2'] = (
        (df['shared_location'] == 'CCunk') 
        & (df['DCA_score'] >= 0.12)
        & (df['RFprob'] >= 0.999)
    )

    df['strategy_3'] = (
        df['in_GI']
        & (df['RFprob'] >= 0.888)
    )

    df['strategy_4'] = (
        df['in_PPIDB']
        & (df['RFprob'] >= 0.681)
    )

    df['strategy_5'] = (
        (df['shared_location'] == 'CCcom') 
        & (df['DCA_score'] >= 0.12)
        & (df['RFprob'] >= 0.3)
        & (df['AFprob'] >= 0.941)
    )

    df['strategy_6'] = (
        (df['shared_location'] == 'CCunk') 
        & (df['DCA_score'] >= 0.12)
        & (df['RFprob'] >= 0.3)
        & (df['AFprob'] >= 0.993)
    )

    df['strategy_7'] = (
        df['in_GI'] 
        & (df['RFprob'] >= 0.25)
        & (df['AFprob'] >= 0.853)
    )

    df['strategy_8'] = (
        df['in_PPIDB']
        & (df['RFprob'] >= 0.25)
        & (df['AFprob'] >= 0.693)
    )

    df['strategy_9'] = (
        df['in_PPIDB']
        & (df['AFprob'] >= 0.926)
    )

    if (~(df['strategy_1']
        | df['strategy_2']
        | df['strategy_3']
        | df['strategy_4']
        | df['strategy_5'] 
        | df['strategy_6']
        | df['strategy_7']
        | df['strategy_8']
        | df['strategy_9'])).any():
        raise UserWarning('some pairs have no associated strategy')

    # implement the hub cutoffs
    af = pd.read_csv('../data/external/Zhang-et-al_biorxiv-2024_AF-scores.txt',
                    sep='\t',
                    names=['pair_id', 'AFprob'])
    if af['pair_id'].duplicated().any():
        raise UserWarning('unexpected duplicates')
    af['uniprot_ac_a'] = af['pair_id'].apply(lambda x: x.split('_')[0])
    af['uniprot_ac_b'] = af['pair_id'].apply(lambda x: x.split('_')[1])
    if (af['uniprot_ac_a'] > af['uniprot_ac_b']).any():
        raise UserWarning('expected IDs to be sorted')
    rf = pd.read_csv('../data/external/Zhang-et-al_biorxiv-2024_RF2-PPI_scores.txt',
                    sep='\t',
                    names=['pair_id', 'RFprob'])
    if rf['pair_id'].duplicated().any():
        raise UserWarning('unexpected duplicates')
    rf['uniprot_ac_a'] = rf['pair_id'].apply(lambda x: x.split('_')[0])
    rf['uniprot_ac_b'] = rf['pair_id'].apply(lambda x: x.split('_')[1])
    if (rf['uniprot_ac_a'] > rf['uniprot_ac_b']).any():
        raise UserWarning('expected IDs to be sorted')

    hub_cutoffs = {
    'strategy_1': (0.8, 26, 0.997),
    'strategy_2': (0.9, 21, 1),
    'strategy_3': (0.8, 46, 0.967),
    'strategy_4': (0.7, 50, 0.967),
    'strategy_5': (0.6, 31, 0.993),
    'strategy_6': (0.75, 36, 1),
    'strategy_7': (0.9, 13, 0.986),
    'strategy_8': (0.9, 22, 0.964),
    'strategy_9': (0.9, 22, 0.964),
    }

    for i_strategy in range(1, 10):
        cutoff_hub, n_partner, cutoff_high = hub_cutoffs[f'strategy_{i_strategy}']
        if i_strategy <= 4:
            score_col = 'RFprob'
            scores = rf
    
        else:
            score_col = 'AFprob'
            scores = af
        deg = degree_per_protein(scores.loc[scores[score_col] > cutoff_hub, :],
                                id_a='uniprot_ac_a',
                                id_b='uniprot_ac_b',
                                )
        hubs = set(deg[deg > n_partner].index.values)
        print(len(hubs))
        df.loc[
            (df['#Protein1'].isin(hubs)
            | df['Protein2'].isin(hubs))
            & (df[score_col] < cutoff_high)
            & (df['CFprob'] < cutoff_high)
            & (df['AFMMprob'] < cutoff_high),
                f'strategy_{i_strategy}'] = False

    df.to_csv(cache_file, sep='\t')

    return df


def load_pubmed_to_date_mapping(in_path):
    in_path = Path(in_path)
    df = pd.read_csv(in_path)
    if df['pmid'].duplicated().any():
        raise UserWarning("Duplicate PMIDs in pmid_to_date")
    df['PubDate_edited'] = (df['best_date']
                            .str.replace('Winter', 'Dec')
                            .str.replace('Fall', 'Sep')
                            .str.replace('Spring', 'Mar')
                            .str.replace('Summer', 'Jun')
                            .str.replace(r'(?P<one>[A-Z][a-z][a-z])-[A-Z][a-z][a-z]',
                                        lambda x: x.group('one'), regex=True)
                            .str.replace(r'(?P<one>[A-Z][a-z][a-z] [0-9]{1,2})-.*',
                                        lambda x: x.group('one'), regex=True)
                            .str.replace(r'(?P<one>[0-9]{1,2})-[0-9]{1,2}',
                                        lambda x: x.group('one'), regex=True))
    if pd.to_datetime(df['PubDate_edited'], errors='coerce').isnull().any():
        print(df.loc[pd.to_datetime(df['PubDate_edited'], errors='coerce').isnull()])
        raise UserWarning("Some dates could not be parsed even after editing")
    df['PubDate_edited'] = pd.to_datetime(df['PubDate_edited'])
    pmid_to_date = df.set_index('pmid')['PubDate_edited']
    return pmid_to_date