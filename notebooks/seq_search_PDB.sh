
cd ../data/external/PDB_aa_seqs
wget https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/pdbaa.gz
gunzip pdbaa.gz
mmseqs createdb pdbaa pdbDB

mmseqs createdb UP000005640_9606.fasta queryDB
mmseqs search queryDB pdbDB resultDB tmp --threads 4 -e 1e-5
mmseqs convertalis queryDB pdbDB resultDB results.m8

mmseqs createdb ../orf_trans_all.fasta yeastqueryDB
mmseqs search yeastqueryDB pdbDB yeastresultDB tmp --threads 4 -e 1e-5
mmseqs convertalis yeastqueryDB pdbDB yeastresultDB yeastresults.m8