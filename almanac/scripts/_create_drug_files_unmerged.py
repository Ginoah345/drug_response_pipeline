import os
import numpy as np
import argparse
import pickle

from src.preprocessing.preprocessing import DrugDatasetPreprocessor
from src.utils.utils import build_char_dict


def create_drug_files(featurization_type):
	"""
	Featurize ALMANAC drugs.

	Parameters
	----------
	featurization_type: str
		The featurization method that will be applied.
	"""
	drug_preprocessor = DrugDatasetPreprocessor(
		dataset_filepath='../data/nci_almanac_preprocessed/drugs/unmerged/SMILES.csv',
		id_col=None,
		smiles_col='SMILES')

	# For TextCNN
	# char_dict, seq_length = build_char_dict('../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv',
	# 										smiles_cols=['SMILES_A', 'SMILES_B'],
	# 										save=False)
	with open('../data/char_dict_seq_len.pkl', 'rb') as f:
		char_dict, seq_length = pickle.load(f)

	# For GCN and GAT
	max_n_atoms_whole_dataset = drug_preprocessor._get_max_number_of_atoms()

	# Featurize drug A and drug B:
	featurizer_options = {'ECFP4': ('ECFPFeaturizer', {'radius': 2, 'length': 1024}, 'ECFP4_1024'),
						  'ECFP6': ('ECFPFeaturizer', {'radius': 3, 'length': 1024}, 'ECFP6_1024'),
						  'LayeredFP': ('LayeredFPFeaturizer', {'fp_size': 1024}, 'LayeredFPFeaturizer'),
						  'GCN': ('GraphFeaturizer', {'zero_pad': True, 'normalize_adj_matrix': True, 'max_num_atoms':max_n_atoms_whole_dataset}, 'GCN_drug'),
						  'GAT': ('GraphFeaturizer', {'zero_pad': True, 'normalize_adj_matrix': False, 'max_num_atoms':max_n_atoms_whole_dataset}, 'GAT_drug'),
						  'MTE': ('MTEmbeddingsFeaturizer', {'embeddings_file':'../data/nci_almanac_preprocessed/drugs/mtembeddings_almanac_smiles.npz'}, 'MTEmbeddingsFeaturizer'),
						  'TextCNN': ('TextCNNFeaturizer', {'char_dict':char_dict, 'seq_length':seq_length}, 'TextCNNFeaturizer')}
	featurizer_opt = featurizer_options[featurization_type]
	drug_preprocessor.featurize(featurizer_opt[0], featurizer_args=featurizer_opt[1],
	                             output_dir='../data/nci_almanac_preprocessed/drugs/unmerged',
	                             output_prefix=featurizer_opt[2], featurize_split_datasets=False,
	                             featurize_full_dataset=True)  # features are automatically saved as .npy files
	
	# create_drug_files
	SMILES = drug_preprocessor.load_data(dataset_filepath='../data/nci_almanac_preprocessed/drugs/unmerged/SMILES.csv')
	smiles_ls = SMILES['SMILES'].tolist()
	if featurizer_opt[0] == 'GraphFeaturizer':
		output_prefix = featurizer_opt[2]
		prefix, dataset_name = output_prefix.split('_')

		output_path_node_features = os.path.join('../data/nci_almanac_preprocessed/drugs/unmerged','{prefix}_nodes_{name}.npy'.format(prefix=output_prefix, name=dataset_name))
		output_path_adjacency_matrices = os.path.join('../data/nci_almanac_preprocessed/drugs/unmerged', '{prefix}_adjmatrix_{name}.npy'.format(prefix=output_prefix, name=dataset_name))

		node_features = np.load(output_path_node_features)
		adjacency_matrices = np.load(output_path_adjacency_matrices)

		output_filepath_node_features = os.path.join('../data/nci_almanac_preprocessed/drugs/unmerged','{prefix}_nodes_{name}.npz'.format(prefix=output_prefix, name=dataset_name))
		output_filepath_adjacency_matrices = os.path.join('../data/nci_almanac_preprocessed/drugs/unmerged', '{prefix}_adjmatrix_{name}.npz'.format(prefix=output_prefix, name=dataset_name))
		np.savez_compressed(output_filepath_node_features, **{smiles: feat for smiles, feat in zip(smiles_ls, node_features)})
		np.savez_compressed(output_filepath_adjacency_matrices, **{smiles: feat for smiles, feat in zip(smiles_ls, adjacency_matrices)})

		os.remove(output_path_node_features)
		os.remove(output_path_adjacency_matrices)
	else:
		output_path_features = os.path.join('../data/nci_almanac_preprocessed/drugs/unmerged', featurizer_opt[2] + '.npy')
		featurized_values = np.load(output_path_features)
		output_filepath = os.path.join('../data/nci_almanac_preprocessed/drugs/unmerged', featurizer_opt[2] + '.npz')
		np.savez_compressed(output_filepath, **{smiles: feat for smiles, feat in zip(smiles_ls, featurized_values)})
		os.remove(output_path_features)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Featurize ALMANAC drugs and save to file')
	parser.add_argument('-t',
	                    '--featurization-type',
	                    type=str,
	                    help='The type of featurization to use')
	args = vars(parser.parse_args())
	print(args)
	create_drug_files(**args)
