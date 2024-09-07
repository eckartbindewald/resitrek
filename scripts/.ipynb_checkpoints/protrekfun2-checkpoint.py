from bioservices import UniProt
import re
import pandas as pd
import ast
import copy
import sys
import numpy as np
import torch
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from utils.foldseek_util import get_struc_seq
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
import os
import pickle

# Load model
config = {
    "protein_config": "weights/ProTrek_650M_UniRef50/esm2_t33_650M_UR50D",
    "text_config": "weights/ProTrek_650M_UniRef50/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_650M_UniRef50/foldseek_t30_150M",
    "load_protein_pretrained": False,
    "load_text_pretrained": False,
    "from_checkpoint": "weights/ProTrek_650M_UniRef50/ProTrek_650M_UniRef50.pt"
}

PARAMS = {
  'outfilename':'functon_matrices_out.pickle',
  'score_thresh1':0.1,
  'score_thresh2':0.01
}

def fetch_protein_sequence_from_uniprot(uniprot_id):
    # Initialize the UniProt service
    u = UniProt()
    
    # Fetch the protein sequence for the given UniProt ID
    result = u.retrieve(uniprot_id, frmt="fasta")
    
    # Extract the sequence part from the FASTA format
    sequence = ''.join(result.split('\n')[1:])
    
    return sequence


def get_structure_sequence(file_path):
    if file_path.endswith('.pdb'):
        ppdb = PandasPdb().read_pdb(file_path)
    elif file_path.endswith('.cif'):
        ppdb = PandasMmcif().read_mmcif(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a PDB or CIF file.")
    
    # Extract sequences for each chain
    sequences = []
    atom_df = ppdb.df['ATOM']
    for chain_id in atom_df['chain_id'].unique():
        chain_df = atom_df[atom_df['chain_id'] == chain_id]
        sequence = chain_df.drop_duplicates(subset=['residue_number', 'residue_name'])['residue_name'].tolist()
        sequences.append(''.join(sequence))
    
    return sequences


def get_structure_sequence_old(filename):
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    sequences = []

    if ext == '.pdb':
        ppdb = PandasPdb().read_pdb(filename)
        chain_ids = ppdb.df['ATOM']['chain_id'].unique()
        for chain in chain_ids:
            seq = ppdb.amino3to1(ppdb.df['ATOM'][ppdb.df['ATOM']['chain_id'] == chain])
            sequences.append(seq)

    elif ext in ['.cif', '.mmcif']:
        pmmcif = PandasMmcif().read_mmcif(filename)
        
        # Check for the correct column name for chain ID
        chain_col = 'auth_asym_id' if 'auth_asym_id' in pmmcif.df['ATOM'].columns else 'label_asym_id'
        
        chain_ids = pmmcif.df['ATOM'][chain_col].unique()
        
        for chain in chain_ids:
            # Use a try-except block to handle potential issues with amino3to1
            try:
                seq = pmmcif.amino3to1(pmmcif.df['ATOM'][pmmcif.df['ATOM'][chain_col] == chain])
            except KeyError:
                # If amino3to1 fails, we'll extract the sequence manually
                residues = pmmcif.df['ATOM'][
                    (pmmcif.df['ATOM'][chain_col] == chain) & 
                    (pmmcif.df['ATOM']['label_atom_id'] == 'CA')
                ]['label_comp_id'].tolist()
                seq = ''.join(PandasMmcif.amino3to1_dict.get(res, 'X') for res in residues)
            
            sequences.append(seq)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return sequences

def read_structure(filename):
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    sequences = []
    if ext == '.pdb':
        ppdb = PandasPdb().read_pdb(filename)
        return ppdb.df['ATOM']
    elif ext in ['.cif', '.mmcif']:
        pmmcif = PandasMmcif().read_mmcif(filename)
        return pmmcif.df['ATOM']
    else:
      raise FileNotFoundError("Extension of structure file has to be .pdb or .cif")


def get_go_annotations(uniprot_id):
    # Initialize UniProt service
    u = UniProt()
    
    # Fetch data for the given UniProt ID
    result = u.search(uniprot_id, columns="id,go")
    
    # print("raw result")
    # print(result)
    result = ast.literal_eval(f"'''{result}'''")
    # print("result after ast:")
    # print(result)
    # Split the result into lines
    lines = result.split("\n")
    # print("raw lines:")
    # print(lines)
    # Check if we have any results
    if len(lines) < 2:
        return []
    
    # The second line contains the data
    data = lines[1].split("\t")
    
    # Check if we have GO annotations
    if len(data) < 2:
        return []
    
    # Parse GO annotations
    go_annotations = data[1].split("; ")
    
    short_names = []
    go_ids = []
    
    for annotation in go_annotations:
        # Use regex to extract GO ID and short name
        # print("annnotation:", annotation)
        match = re.search(r'(.+?)\s+\[(GO:\d+)\]', annotation)
        if match:
            short_name, go_id = match.groups()
            short_names.append(short_name.strip())
            go_ids.append(go_id)

    # Create DataFrame
    # df = pd.DataFrame(data={
    #    'Short_Name': short_names,
    #    'GO_ID':go_ids}).reset_index(drop=True)

    return short_names, go_ids

AA = 'ACDEFGHIKLMNPQRSTVWY'

def sample_function_influence_at_pos(
    model, sequence, description, pos):
    """
    For a sequence and model and function (a text description)
    """
    fun_scores = {}
    seq_orig = copy.deepcopy(sequence)
    text_embedding = model.get_text_repr([description])
    curr_aa = sequence[pos]
    # print("Current amino acid for position", pos, ":")
    # print(curr_aa)
    # print(AA, curr_aa in AA)
    assert curr_aa in AA
    higher_eq_count = 0
    for aa in AA:
        sq = copy.deepcopy(seq_orig)
        sq = sq[:pos] + aa + sq[(pos+1):]
        seq_embedding = model.get_protein_repr([sq])
        # Calculate similarity score between protein sequence and text
        seq_text_score = seq_embedding @ text_embedding.T / model.temperature
        fun_scores[aa] = seq_text_score.item()
    # print("Defined scores for position", pos,":")
    curr_score = fun_scores[curr_aa]
    for aa in AA:
        if fun_scores[aa] >= curr_score:
            higher_eq_count+=1
    return higher_eq_count / len(fun_scores) # simple "P-value"


def sample_function_influence_at_pos_pair(
    model, sequence, description, pos, pos2,
    visited_min=10, early_stop_thresh=0.2):
    """
    For a sequence and model and function (a text description)
    """
    fun_scores = {}
    # seq_orig = copy.deepcopy(sequence)
    text_embedding = model.get_text_repr([description])
    curr_aa = sequence[pos]
    curr_aa2 = sequence[pos2]
    # print("Current amino acid for position", pos, ":")
    # print(curr_aa)
    # print(AA, curr_aa in AA)
    assert curr_aa in AA
    higher_eq_count = 0
    curr_score = None
    visited = 0
    # visited current / native amino acids first:
    AAmod = curr_aa + AA.replace(curr_aa,'')
    AAmod2 = curr_aa2 + AA.replace(curr_aa2, '')
    for aa in AAmod:
        for aa2 in AAmod2:
          # sq = copy.deepcopy(seq_orig)
          sql = list(sequence)
          sql[pos] = aa
          sql[pos2] = aa2
          sq = ''.join(sql)
          seq_embedding = model.get_protein_repr([sq])
          # Calculate similarity score between protein sequence and text
          seq_text_score = seq_embedding @ text_embedding.T / model.temperature
          fun_scores[aa + aa2] = seq_text_score.item()
          if aa == curr_aa and aa2 == curr_aa2:
              curr_score = fun_scores[aa + aa2]
          assert curr_score is not None
          if fun_scores[aa + aa2] >= curr_score:
              higher_eq_count+=1
          visited += 1
          if (visited >= visited_min) and ((higher_eq_count / visited) > early_stop_thresh):
              return higher_eq_count / visited
    # print("Defined scores for position", pos,":")
    return higher_eq_count / len(fun_scores) # simple "P-value"


def sample_function_influence(model, sequence, description):
    n = len(sequence)
    result = [0]*n
    print("Analyzing description:")
    print(description)
    print("Position\tResidue\tScore")
    for pos in range(n):
        result[pos] =  sample_function_influence_at_pos(
            model, sequence, description, pos
        )
        print(pos+1, sequence[pos], result[pos], sep='\t')
    return result


def calculate_ca_distance(atom_df, residue_id1, residue_id2):
    # Filter for C-alpha atoms of the specified residues in zero-based counting
    ca1 = atom_df[(atom_df['atom_name'] == 'CA') & (atom_df['residue_number'] == (residue_id1+1))]
    ca2 = atom_df[(atom_df['atom_name'] == 'CA') & (atom_df['residue_number'] == (residue_id2+1))]
    
    # Check if both C-alpha atoms are found
    if ca1.empty or ca2.empty:
        raise ValueError(f"C-alpha atom not found for residue {residue_id1 if ca1.empty else residue_id2}")
    
    # Extract coordinates
    coords1 = ca1[['x_coord', 'y_coord', 'z_coord']].values[0]
    coords2 = ca2[['x_coord', 'y_coord', 'z_coord']].values[0]
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(coords1 - coords2)
    
    return distance


def sample_function_pair_influence(model, sequence, description,
    structure, dist_max=10.0, interesting=None, result=None):
    n = len(sequence)
    if result is None:
        result = np.zeros([n,n])
    print("Analyzing description:")
    print(description)
    print("Position\tResidue\tScore")
    for pos in range(n-1):
        for pos2 in range(pos+1,n):
            if interesting is not None:
                if pos not in interesting and pos2 not in interesting:
                    continue
            if result[pos, pos2] != 0.0:
                continue # already visited
            dist = calculate_ca_distance(structure, pos, pos2)
            if dist > dist_max:
                continue
            result[pos, pos2] =  sample_function_influence_at_pos_pair(
                model, sequence, description, pos, pos2)
            result[pos2, pos] = result[pos, pos2]
            print(sequence[pos], pos+1, sequence[pos2], pos2+1, ":", round(result[pos, pos2],4), round(dist,3))
    return result



# Load protein and text
 # , ["A"])["A"]

# Example usage

if len(sys.argv) < 2:
    print("Usage: unprofun2.py <UNIPROTID>")
    sys.exit(0)


def main(argv=sys.argv, outfilename=PARAMS['outfilename'],
    score_thresh1=PARAMS['score_thresh1'],
    score_thresh2=PARAMS['score_thresh2']
    ):
    uniprot_id = argv[1]
    other_go = []
    if len(argv) > 3:
      other_go = argv[3:]
    structure_file = argv[2]
    # sequence = get_structure_sequence(filename)[0]
    structure = read_structure(structure_file)
    print(structure.head().columns)
    print(structure.head().to_string())
    sequence = fetch_protein_sequence_from_uniprot(uniprot_id)
    print("sequence:")
    print(sequence)
    descriptions, go_ids = get_go_annotations(uniprot_id)
    print("Descriptions:")
    print(descriptions)
    print("GO ids:")
    print(go_ids)
    print("Initializing model...")
    device = "cuda"
    model = ProTrekTrimodalModel(**config).eval().to(device)
    print("done.")
    m = len(descriptions)
    n = len(sequence)
    result1 = {}
    result2 = {}
    for i in range(m):
        mtx = np.zeros([n,n])
        go_id = go_ids[i] # df.at[i, 'GO_ID']
        if len(other_go) > 0 and go_id not in other_go:
            continue
        description = descriptions[i] # df.at[i, 'Short_Name']
        print("Working on descriptio in row", i, ":", description)
        result1[go_id] = sample_function_influence(model, sequence,
            description)
        interesting = []
        for j in range(len(result1[go_id])):
            if result1[go_id][j] <= score_thresh1: # like P-value, so smaller is better
                interesting.append(j)
        print("1D scan for", description, ":")
        print(result1[go_id])
        print("Interesting residues:", list(pd.Series(interesting)+1))
        result2[go_id] = sample_function_pair_influence(model, sequence,
            description, structure, interesting=interesting)
        for j in range(result2[go_id].shape[0]):
            for k in range(result2[go_id].shape[1]):
                if result2[go_id][j][k] <= score_thresh2: # like P-value, so smaller is better
                    interesting.append(j)
                    interesting.append(k)
        interesting = list(set(interesting))
        print("Interesting residues after first 2d scane:", list(pd.Series(interesting)+1))
        result2[go_id] = sample_function_pair_influence(model, sequence,
            description, structure, interesting=interesting, result=result2[go_id])
        print("Writing to output file", outfilename)
        with open(outfilename, 'wb') as file:
            pickle.dump({'result1':result1, 'result2':result2}, file)


if __name__ == "__main__":
    print("called uniprofun2.py with", sys.argv)
    argv = [s for s in sys.argv if s.strip()] # remove empty strings
    main(argv)

