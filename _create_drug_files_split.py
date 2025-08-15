import numpy as np
import pandas as pd
import argparse
import pickle
import os

class create_drug_files(object):
    def __init__(self, 
                 response_file='../data/nci_almanac_preprocessed/response/almanac_cellminercdb_with_preprocessed_smiles_no_duplicate_triples.csv', 
                 embeddings_file='../data/nci_almanac_preprocessed/drugs/unmerged/ECFP4_1024.npz', 
                 smiles_col='SMILES_A',
                 split_file='../data/splits/train_val_test_groups_split_inds_12321.pkl', 
                 split_dataset=True, 
                 full_dataset=False):
        """
        Parameters
        ----------
        response_file: str
            CSV 文件，包含 drug SMILES 和响应值
        embeddings_file: str
            npz 文件，key 是 SMILES，value 是 embedding (numpy array)
        smiles_col: str
            表示药物 SMILES 的列名
        response_col: str
            响应值的列名（可换成别的 drug 对应的响应列）
        split_file: str or None
            pkl 文件，保存 train/val/test 的索引列表
        full_dataset: bool
            True 时不划分数据集，只返回全部数据
        """
        # 加载 response
        if response_file.endswith('.csv'):
            self.response_df = pd.read_csv(response_file)
        else:
            raise ValueError("Unsupported response file format.")

        # 加载 embeddings
        self.embeddings = np.load(embeddings_file)

        self.smiles_col = smiles_col
        self.split_dataset = split_dataset
        self.full_dataset = full_dataset

        if split_dataset:
            if split_file is None:
                raise ValueError("split_file must be provided when full_dataset=False.")
            with open(split_file, 'rb') as f:
                self.split_indices = pickle.load(f)
            self.dataset_names = ['train', 'val', 'test'] if len(self.split_indices) == 3 else ['train', 'test']

    def _get_embedding(self, smiles):
        """根据 SMILES 获取 embedding（平均展平）"""
        emb = self.embeddings[smiles]
        return emb

    def get_dataset(self):
        """
        full_dataset=True → 返回 X
        full_dataset=False → 返回 {'train': X, 'val': ..., 'test': ...}
        """
        if self.full_dataset:
            featurized_arr = np.stack([self._get_embedding(s) for s in self.response_df[self.smiles_col]])
            return featurized_arr
        
        if self.split_dataset:
            split_dataset = {}
            for name, ind_list in zip(self.dataset_names, self.split_indices):
                response_df_split = self.response_df.iloc[ind_list]
                featurized_arr = np.stack([self._get_embedding(s) for s in response_df_split[self.smiles_col]])
                split_dataset[name] = featurized_arr
            return split_dataset
        
    def save_dataset(self, output_dir='../data/nci_almanac_preprocessed/drugs/split', output_prefix='ECFP4_1024_drugA'):
        os.makedirs(output_dir, exist_ok=True)
        if self.full_dataset:
            dataset = self.get_dataset()
            save_path = os.path.join(output_dir, f"{output_prefix}.npy")
            np.save(save_path, dataset)
        
        if self.split_dataset:
            split_dataset = self.get_dataset()
            for split_name, dataset in split_dataset.items():
                save_path = os.path.join(output_dir, f"{output_prefix}_{split_name}.npy")
                np.save(save_path, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Featurize drug dataset and save to file')
    parser.add_argument('--response-file', type=str, help='Path to response CSV file')
    parser.add_argument('--embeddings-file', type=str, help='Path to embeddings npz file')
    parser.add_argument('--smiles-col', type=str, help='Name of the SMILES column')
    parser.add_argument('--split-file', type=str, default=None, help='Path to pickle split indices file')
    parser.add_argument('--split-dataset', action='store_true', help='If set, split dataset')
    parser.add_argument('--full-dataset', action='store_true', help='If set, use full dataset')
    parser.add_argument('--output-dir', type=str, help='Directory to save output')
    parser.add_argument('--output-prefix', type=str, help='Prefix for saved files')

    args = parser.parse_args()
    args_dict = vars(args)
    dataset = create_drug_files(
        response_file=args_dict['response_file'],
        embeddings_file=args_dict['embeddings_file'],
        smiles_col=args_dict['smiles_col'],
        split_file=args_dict.get('split_file'),
        split_dataset=args_dict.get('split_dataset', False),
        full_dataset=args_dict.get('full_dataset', False)
    )
    dataset.save_dataset(
        output_dir=args_dict['output_dir'],
        output_prefix=args_dict['output_prefix']
    )