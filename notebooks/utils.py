import math
import os
import sys
import warnings
import glob
import hashlib
import base64

import numpy as np
import pandas as pd
import networkx as nx
import igraph
from tqdm import tqdm

from datasets import load_pdb_id_to_date


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


def partners_per_protein(df, id_a="orf_id_a", id_b="orf_id_b"):
    """For every protein the proteins that it interacts with
    Args:
        df (DataFrame): PPI data, one row per pair.
    Returns:
        Series: index: ORF ID, values: set of ORF IDs of interaction partners
    """
    p = pd.merge(
        df.groupby(id_a).agg({id_b: set}),
        df.groupby(id_b).agg({id_a: set}),
        how="outer",
        left_index=True,
        right_index=True,
    )

    def union_columns(row):
        if pd.isnull(row[id_b]):
            return row[id_a]
        elif pd.isnull(row[id_a]):
            return row[id_b]
        else:
            return row[id_a].union(row[id_b])

    out = p.apply(union_columns, axis=1)
    out.name = "partners"
    return out


def _guess_id_type_and_suffixes(columns):
    id_a, id_b = columns[:2]
    if id_a.split('_')[:-1] == id_b.split('_')[:-1]:
        return ('_'.join(id_a.split('_')[:-1]),
                ('_' + id_a.split('_')[-1],
                 '_' + id_b.split('_')[-1]))
    else:
        raise UserWarning('Failed to guess id_type and suffixes. Please specify.')


def _guess_node_id(columns, suffixes=('_a', '_b')):
    """
    Given a table of edges, try and guess the node IDs.

    Should I give a warning? i.e. guessing gene id type is x, to silence pass id_type=x?
    Maybe only give warning if it's ambiguous? i.e. there are not multiple?

    """
    cols_a = [c[:-len(suffixes[0])] for c in columns if c.endswith(suffixes[0])]
    cols_b = [c[:-len(suffixes[1])] for c in columns if c.endswith(suffixes[1])]
    for col_a in cols_a:
        for col_b in cols_b:
            if col_a == col_b:
                return col_a
    raise UserWarning('Could not guess node IDs from: ' + ' | '.join(columns))


def merge_node_and_edge_tables(nodes,
                               edges,
                               id_type=None,
                               suffixes=('_a', '_b'),
                               node_id_column=None):
    """Combine data on nodes and edges into a table of edges.

    Args:
        nodes (pandas.DataFrame): table of nodes
        edges (pandas.DataFrame): table of edges


    """
    if id_type is None:
        _guess_node_id(edges.columns, suffixes)
    if node_id_column is None:
        df = pd.merge(edges,
                      nodes,
                      right_index=True,
                      left_on=id_type + suffixes[0],
                      how='left')
        df = pd.merge(df,
                      nodes,
                      right_index=True,
                      left_on=id_type + suffixes[1],
                      how='left',
                      suffixes=suffixes)
    else:
        df = pd.merge(edges,
                      nodes,
                      right_on=node_id_column,
                      left_on=id_type + suffixes[0],
                      how='left')
        df = pd.merge(df,
                      nodes,
                      right_on=node_id_column,
                      left_on=id_type + suffixes[1],
                      how='left',
                      suffixes=suffixes)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def split_node_and_edge_tables(edges, id_type=None, suffixes=None):
    es = edges.copy()
    if id_type is None:
        if suffixes is None:
            id_type, suffixes = _guess_id_type_and_suffixes(es.columns)
        else:
            id_type = _guess_node_id(es.columns, suffixes)
    elif suffixes is None:
        suffixes = es.columns[0].replace(id_type, ''), es.columns[1].replace(id_type, '')
    columns_a = [c for c in es.columns if c.endswith(suffixes[0])]
    columns_b = [c for c in es.columns if c.endswith(suffixes[1])]    
    ns_a = (es.loc[:, columns_a]
              .rename(columns={c: c[:-len(suffixes[0])] for c in columns_a})
              .drop_duplicates())
    ns_b = (es.loc[:, columns_b]
              .rename(columns={c: c[:-len(suffixes[1])] for c in columns_b})
              .drop_duplicates())
    ns = (pd.concat([ns_a, ns_b])
            .drop_duplicates()
            .set_index(id_type, verify_integrity=True))
    gene_id_columns = [id_type + s for s in suffixes]
    es = es.loc[:, gene_id_columns + [c for c in es.columns if c not in (columns_a + columns_b)]]
    return ns, es


def format_network(data, fmt, id_type=None, suffixes=('_a', '_b'), directed=False):
    """Convert data format of network between pandas/networkx/igraph etc.

    fmt options are:

        - `pandas`: DataFrame
        - `nx`: networkx Graph
        - `igraph`: igraph Graph
        - `list`: list of a dict for each PPI

    Args:
        data (pandas.DataFrame / networkx.Graph / igraph.Graph / list): input network
        fmt (str): format to convert to; pandas/nx/igraph/list
        id_type (str): gene/protein identifier type used
        suffixes (tuple(str, str)): at the end of id_type to distinguish the two nodes
        directed (bool): return directed for igraph/networkx

    Returns:
        (pandas.DataFrame / networkx.Graph / igraph.Graph / list): network in specified format

    """
    valid_fmt = ['pandas', 'nx', 'igraph', 'list']
    if fmt not in valid_fmt:
        raise UserWarning('Unsupported fmt: ' + fmt +
                          '\nValid options are: ' + '/'.join(valid_fmt))
    fmt_out = fmt
    fmt_in = 'unknown'
    for fmt_desc, data_type in [('pandas', pd.DataFrame),
                                ('nx', nx.Graph),
                                ('igraph', igraph.Graph),
                                ('list', list)]:
        if isinstance(data, data_type):
            fmt_in = fmt_desc
            break
    if fmt_in == 'unknown':
        raise ValueError('Unsupported input type: ' + type(data))
    if id_type is None:
        if fmt_in == 'pandas':
            id_type, suffixes = _guess_id_type_and_suffixes(data.columns)
        else:
            raise ValueError('Require value for id_type argument')
    id_a = id_type + suffixes[0]
    id_b = id_type + suffixes[1]

    if fmt_in != 'pandas' and fmt_out != 'pandas':
        # via pandas.DataFrame, so don't have to code every possible conversion
        tbl = format_network(data,
                             'pandas',
                             id_type=id_type,
                             suffixes=suffixes,
                             directed=directed)
        return format_network(tbl,
                              fmt_out,
                              id_type=id_type,
                              suffixes=suffixes,
                              directed=directed)

    elif fmt_in == 'pandas' and fmt_out == 'pandas':
        return data

    elif fmt_in == 'pandas' and fmt_out == 'nx':
        if directed:
            graph_type = nx.DiGraph()
        else:
            graph_type = nx.Graph()
        node_df, edge_df = split_node_and_edge_tables(data, id_type=id_type, suffixes=suffixes)
        edge_attr_cols = [c for c in edge_df.columns if c not in (id_a, id_b)]
        if len(edge_attr_cols) == 0:
            edge_attr_cols = None
        nw = nx.from_pandas_edgelist(edge_df,
                                     source=id_a,
                                     target=id_b,
                                     edge_attr=edge_attr_cols,
                                     create_using=graph_type)
        for n in nw.nodes:
            for c in node_df.columns:
                nw.nodes[n][c] = node_df[c][n]
        return nw

    elif fmt_in == 'pandas' and fmt_out == 'igraph':
        node_df, edge_df = split_node_and_edge_tables(data, id_type=id_type, suffixes=suffixes)
        g = igraph.Graph()
        g = g.TupleList([(a, b) for a, b in edge_df[[id_a, id_b]].values],
                        directed=directed)
        for column in edge_df.columns:
            if column not in [id_a, id_b]:
                g.es[column] = edge_df[column].values
        for column in node_df.columns:
            g.vs[column] = node_df.loc[g.vs['name'], column].values
        return g

    elif fmt_in == 'pandas' and fmt_out == 'list':
        d = data.to_dict()
        mylist = [{k: d[k][idx] for k in d.keys()} for idx in d[id_a].keys()]
        return mylist

    elif fmt_in == 'nx' and fmt_out == 'pandas':
        out = nx.to_pandas_edgelist(data).rename(columns={'source': id_a,
                                                        'target': id_b})
        out = out.set_index(out[id_a] + '_' + out[id_b])
        node_table = pd.DataFrame({n: data.nodes[n] for n in data.nodes}).T
        out = merge_node_and_edge_tables(node_table, out, id_type=id_type, suffixes=suffixes)
        return out

    elif fmt_in == 'igraph' and fmt_out == 'pandas':
        d = {id_a: [data.vs[e.source]['name'] for e in data.es],
             id_b: [data.vs[e.target]['name'] for e in data.es]}
        d.update({k: data.es[k] for k in data.es.attribute_names()})
        edge_df = pd.DataFrame(d)
        edge_df = edge_df.set_index(edge_df[id_a] + '_' + edge_df[id_b])
        node_df = pd.DataFrame({k: data.vs[k] for k in data.vs.attribute_names()}).set_index('name')
        out = merge_node_and_edge_tables(node_df, edge_df, id_type=id_type, suffixes=suffixes)
        return out

    elif fmt_in == 'list' and fmt_out == 'pandas':
        out = pd.DataFrame(data)
        out = out.set_index(out[id_a] + '_' + out[id_b])
        return out

    else:
        raise UserWarning('Something went wrong...')


def log_output(s):
    print(s)
    with open("../output/log.txt", "a") as f:
        f.write(s + "\n")


def round_down_sigfig(num, sig_figs):
    if num == 0:
        return 0
    order_of_magnitude = math.floor(math.log10(abs(num)))
    scale = 10 ** (order_of_magnitude - sig_figs + 1)
    return math.floor(num / scale) * scale


def _hit_in_PDB(orf_name_a, orf_name_b, algn):
    hits = algn.loc[algn['query'].isin([orf_name_a, orf_name_b]), ['query', 'PDB_ID', 'chain_ID']]
    if not ((hits['query'] == orf_name_a).sum() > 0) and ((hits['query'] == orf_name_b).sum() > 0):
        return False
    return (((hits.groupby('PDB_ID')['query'].nunique() == 2) 
            & (hits.groupby('PDB_ID')['chain_ID'].nunique() >= 2)).sum() > 0)


def has_interolog_structure_yeast(df, col_a='orf_name_a', col_b='orf_name_b', date_cutoff=None):
    m8_columns = ['query',
                  'target',
                  'pident',
                  'alnlen',
                  'mismatches',
                  'gapopens',
                  'qstart',
                  'qend',
                  'tstart',
                  'tend',
                  'evalue',
                  'bits']
    algn = pd.read_csv('../data/external/PDB_aa_seqs/yeastresults.m8', 
                       names=m8_columns,
                       sep='\t')
    algn['PDB_ID'] = algn['target'].apply(lambda x: x.split('_')[0])
    algn['chain_ID'] = algn['target'].apply(lambda x: x.split('_')[1])
    if date_cutoff is not None:
        pdb_id_to_date = load_pdb_id_to_date(algn['PDB_ID'].unique())
        algn['date'] = algn['PDB_ID'].str.lower().map(pdb_id_to_date)
        if algn['date'].isnull().any():
            raise UserWarning('missing values')
        algn = algn.loc[algn['date'] < date_cutoff, :]
    return df.apply(lambda x: _hit_in_PDB(x[col_a], x[col_b], algn), axis=1)


def ccsblib_config_dir():
    """Directory ccsblib looks for config files

    Returns:
        str: path

    """
    home_dir = os.path.expanduser('~')
    home_dir = os.path.realpath(home_dir)  # in case symlink
    if 'CCSBLIB_CONFIG_DIR' in os.environ:
        config_dir = os.environ['CCSBLIB_CONFIG_DIR']
    else:
        config_dir = os.path.join(home_dir, '.ccsblib')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return config_dir


def ccsblib_cache_dir():
    """Directory ccsblib saves data files for later use.

    Returns:
        str: path

    """
    if 'CCSBLIB_CACHE' in os.environ:
        return os.environ['CCSBLIB_CACHE']
    home_dir = os.path.expanduser('~')
    home_dir = os.path.realpath(home_dir)  # in case symlink
    if sys.platform == 'darwin':
        return os.path.join(home_dir, 'Library', 'ccsblib')
    if os.name == 'nt':
        appdata = os.environ.get('APPDATA', None)
        if appdata:
            return os.path.join(appdata, 'ccsblib')
        return os.path.join(ccsblib_config_dir(), 'data')
    else:
        # Linux, non-OS X Unix, AIX, etc.
        xdg = os.environ.get('XDG_DATA_HOME', None)
        if not xdg:
            xdg = os.path.join(home_dir, '.local', 'share')
        return os.path.join(xdg, 'ccsblib')


def generate_random_networks(real_nw, n=1000, cache=True):
    """Generate a set of degree-preserved randomized networks.

    The networks produced will be saved for re-use later.

    Warning:
        The produced networks will be a single fully connected component.
        You should check that these will be representative of your input
        network.

    Args:
        real_nw (igraph.Graph): Input network to randomize. Must have no
                                self-interactions
        n (int): Number of random networks to generate
        cache (bool): Controls whether to save the networks for re-use later

    Returns:
        list(igraph.Graph): randomly generated networks

    """
    if any(real_nw.is_loop()):
        msg = """real_nw must not contain self-interactions.
                 Remove with:
                 g.simplify()
                 g.delete_vertices([v for v in g.vs if g.degree() == 0])"""
        raise ValueError(msg)
    if any([d == 0 for d in real_nw.degree()]):
        msg = """real_nw must not contain any unconnected nodes.
                 Remove with:
                 g.delete_vertices([v for v in g.vs if v.degree() == 0])"""
        raise ValueError(msg)
    cache_dir = ccsblib_cache_dir()
    degree_seq = sorted(real_nw.degree())
    ds_hash = base64.urlsafe_b64encode(hashlib.md5(repr(degree_seq).encode('utf8')).digest()).decode('utf8')
    rand_nws = []
    i = 1
    while True:
        cache_dir = os.path.join(cache_dir,
                                 'random_networks',
                                 ds_hash + '_' + str(i))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            break
        else:
            rand_nws = [igraph.load(f) for f in glob.glob(cache_dir + '/random_network_*.txt')]
            if all([sorted(nw.degree()) == degree_seq for nw in rand_nws]):
                break
            else:
                i += 1
    if len(rand_nws) < n:
        for __ in tqdm(range(n - len(rand_nws))):
            rand_nws.append(igraph.Graph.Degree_Sequence(real_nw.degree(),
                                                         method='vl'))
    if cache:
        for i, rand_nw in enumerate(rand_nws):
            outpath = os.path.join(cache_dir,
                                   'random_network_' + str(i) + '.txt')
            rand_nw.write_edgelist(outpath)
    if all([real_nw.degree() == rand_nw.degree() for rand_nw in rand_nws]):
        for attribute in real_nw.vertex_attributes():
            for rand_nw in rand_nws:
                rand_nw.vs[attribute] = real_nw.vs[attribute]
    else:
        warnings.warn('WARNING: random network nodes in different order to real network')

        def get_degree(v):
            return v.degree()

        for attribute in real_nw.vertex_attributes():
            for rand_nw in rand_nws:
                for v_real, v_rand in zip(sorted(real_nw.vs, key=get_degree),
                                          sorted(rand_nw.vs, key=get_degree)):
                    v_rand[attribute] = v_real[attribute]
    return rand_nws
