"""This file contains the embedding dict class. This class is used to dynamically store the embeddings, starting from
an overall list of all the embeddings for all the proteins, and then fishing out the embeddings for the proteins of interest and 
dumping the rest of the embeddings to be space efficient. This class is inspired by the old producing_d_embeddings.ipynb file"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

class EmbeddingDict:
    """
    A class to handle protein embeddings, including loading, filtering, and processing
    embeddings for proteins of interest while maintaining memory efficiency.
    """
    
    def __init__(self, embedding_file_path: str):
        """
        Initialize the EmbeddingDict.
        
        Args:
            embedding_file_path: Path to the embedding file (TSV or CSV format)
        """
        self.embedding_file_path = embedding_file_path
        self.embeddings_df = None
        self.filtered_embeddings = None
        self.protein_mapping = {}
        self.missing_proteins = []
        self.found_proteins = []
        self.similarity_matrix = None
        self.connection_matrix = None
        
    def load_embeddings(self, low_memory: bool = False) -> pd.DataFrame:
        """
        Load embeddings from the file.
        
        Args:
            low_memory: If True, uses low_memory=False in pandas to handle mixed types
            
        Returns:
            DataFrame containing the loaded embeddings
        """
        try:
            # Determine file format and load accordingly
            if self.embedding_file_path.endswith('.txt') or self.embedding_file_path.endswith('.tsv'):
                if low_memory:
                    self.embeddings_df = pd.read_csv(self.embedding_file_path, sep='\t', low_memory=False)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.embeddings_df = pd.read_csv(self.embedding_file_path, sep='\t')
            else:
                if low_memory:
                    self.embeddings_df = pd.read_csv(self.embedding_file_path, low_memory=False)
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.embeddings_df = pd.read_csv(self.embedding_file_path)
            
            print(f"Loaded embeddings with shape: {self.embeddings_df.shape}")
            return self.embeddings_df
            
        except Exception as e:
            raise Exception(f"Error loading embeddings from {self.embedding_file_path}: {str(e)}")
    
    def restructure_bionic_embeddings(self):
        """used specifically for bionic embeddings to get into a common format which is column names=proteins, rows=embedding values
        position by position"""
        self.embeddings_df.rename(columns={'Unnamed: 0':'Proteins'},inplace=True)
        flipped=self.embeddings_df.transpose()
        #set the column headings to row 1:
        flipped.columns=flipped.iloc[0]
        flipped.drop("Proteins", inplace=True)
        del self.embeddings_df
        self.embeddings_df=flipped
        
        return self.embeddings_df
    
  
    def rename_col_HGNC_to_UniProt(self, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Convert column names from HGNC symbols to UniProt IDs.
        
        Args:
            mapping: Optional mapping dictionary. If None, uses self.protein_mapping
            
        Returns:
            DataFrame with converted column names
        """
        if self.embeddings_df is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
        if mapping is None:
            mapping = self.protein_mapping

        if not mapping:
            raise ValueError("No mapping available. add mapping file into object or provide mapping.")

        # Create a copy for conversion
        converted_df = self.embeddings_df.copy()
        
        # Build the complete renaming dictionary first
        rename_dict = {}
        conversion_count = 0
        unmapped_count = 0
        
        for col_name in converted_df.columns:
            if col_name in mapping:
                rename_dict[col_name] = mapping[col_name]
                conversion_count += 1
            else:
                unmapped_count += 1
        
        # Perform all renaming in a single operation
        converted_df.rename(columns=rename_dict, inplace=True)

        print(f"Converted {conversion_count} column names to UniProt IDs")
        print(f"{unmapped_count} column names could not be mapped to UniProt IDs")
        self.embeddings_df = converted_df
        
        return converted_df
    

    def load_protein_list(self, protein_file_path: str, protein_column: int = 1, 
                         header: Optional[int] = None) -> List[str]:
        """
        Load list of proteins of interest from a CSV file.
        
        Args:
            protein_file_path: Path to CSV file containing protein IDs
            protein_column: Column index containing protein IDs (0-based)
            header: Header row index, None if no header
            
        Returns:
            List of protein IDs
        """
        try:
            df = pd.read_csv(protein_file_path, header=header)
            proteins = df.iloc[:, protein_column].tolist()
            # Clean protein IDs: strip whitespace and convert to uppercase
            proteins = [str(prot).strip().upper() for prot in proteins if pd.notna(prot)]
            print(f"Loaded {len(proteins)} proteins of interest")
            return proteins
            
        except Exception as e:
            raise Exception(f"Error loading protein list from {protein_file_path}: {str(e)}")


    def filter_embeddings_for_proteins(self, proteins_of_interest: List[str]=None,
                                     fill_missing_with_zeros: bool = True) -> pd.DataFrame:
        """
        Filter embeddings to include only proteins of interest, maintaining order.
        Handles duplicate proteins by adding suffixes.
        
        Args:
            proteins_of_interest: List of protein IDs to extract
            fill_missing_with_zeros: If True, fill missing proteins with zeros
            
        Returns:
            DataFrame with filtered embeddings
        """
        if self.embeddings_df is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
        # Track protein occurrences for handling duplicates
        protein_counts = {}
        self.missing_proteins = []
        self.found_proteins = []
        
        # Create new DataFrame with same number of rows as original
        self.filtered_embeddings = pd.DataFrame(index=self.embeddings_df.index)
        
        for protein in proteins_of_interest:
            # Handle duplicate proteins by adding suffixes
            if protein not in protein_counts:
                protein_counts[protein] = 1
                column_name = protein
            else:
                protein_counts[protein] += 1
                column_name = f"{protein}_{protein_counts[protein]}"
            
            # Check if protein exists in embeddings
            if protein in self.embeddings_df.columns:
                self.filtered_embeddings[column_name] = self.embeddings_df[protein]
                self.found_proteins.append(protein)
            else:
                if fill_missing_with_zeros:
                    # Fill with zeros for missing proteins
                    self.filtered_embeddings[column_name] = 0
                self.missing_proteins.append(protein)
        
        # Remove duplicates from missing/found lists for reporting
        unique_missing = list(set(self.missing_proteins))
        unique_found = list(set(self.found_proteins))
        
        print(f"Number of proteins found in embeddings: {len(unique_found)}")
        print(f"Number of proteins not found in embeddings: {len(unique_missing)}")
        
        if unique_missing:
            print(f"Missing proteins: {unique_missing}")
        
        return self.filtered_embeddings


    def generate_similarity_matrix(self, similarity_metric: str = 'cosine', normalize: bool = True, absolute: bool = False,fill_missing_with_random: bool = False) -> pd.DataFrame:
        """
        Generate a similarity matrix for the embeddings.
        """
        print('non-normed and other similarity metrics not implemented yet')
        if self.filtered_embeddings is None:
            raise ValueError("No filtered embeddings available. Call filter_embeddings_for_proteins() first.")
        embeddings = self.filtered_embeddings.values.astype(np.float32)
        
        if similarity_metric == 'cosine':
            normed_embeddings=embeddings/np.linalg.norm(embeddings,axis=0,keepdims=True)
            similarity_matrix=np.dot(normed_embeddings.T, normed_embeddings)
            similarity_matrix = pd.DataFrame(
                similarity_matrix,
                index=self.filtered_embeddings.columns,
                columns=self.filtered_embeddings.columns
            )
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

        if normalize:
            print('not yet implemented')
        if fill_missing_with_random:
            mean_val = np.nanmean(similarity_matrix.values)
            std_val = np.nanstd(similarity_matrix.values)
            nan_mask = np.isnan(similarity_matrix.values)
            random_fill = np.random.normal(loc=mean_val, scale=std_val, size=similarity_matrix.shape)
            similarity_matrix.values[nan_mask] = random_fill[nan_mask]


        if absolute:    
            similarity_matrix=np.abs(similarity_matrix)

        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    

    def generate_connection_matrix(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate a connection matrix for the embeddings.
        """
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix available. Call generate_similarity_matrix() first.")
        self.connection_matrix = self.similarity_matrix.applymap(lambda x: 1 if x >= threshold else 0)
        return self.connection_matrix
    

    def save_matrices(self, output_path: str) -> None:
        """
        Save the similarity and connection matrices to a CSV file.
        """
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix available. Call generate_similarity_matrix() first.")
        if self.connection_matrix is None:
            raise ValueError("No connection matrix available. Call generate_connection_matrix() first.")
        if self.filtered_embeddings is None:
            raise ValueError("No filtered embeddings available. Call filter_embeddings_for_proteins() first.")
        
        self.filtered_embeddings.to_csv(output_path + '_filtered_embeddings.csv',index=False,header=False)
        self.similarity_matrix.to_csv(output_path + '_similarity_matrix.csv',index=False,header=False)
        self.connection_matrix.to_csv(output_path + '_connection_matrix.csv',index=False,header=False)
    

    def pad_matrix(self, matrix: pd.DataFrame, target_x: int, target_y: int, mean: float = 0.01, std: float = 0.2) -> pd.DataFrame:
        """
        Pad a matrix to the target (x, y) dimensions, filling new values with random values
        from a normal distribution (mean=0.01, std=1).
        """
        import numpy as np
        import pandas as pd

        current_x, current_y = matrix.shape

        # Pad rows if needed
        if current_x < target_x:
            num_new_rows = target_x - current_x
            # Generate random values for new rows
            new_rows = np.random.normal(loc=mean, scale=std, size=(num_new_rows, current_y))
            new_rows_df = pd.DataFrame(new_rows, columns=matrix.columns)
            matrix = pd.concat([matrix, new_rows_df], ignore_index=True)

        # Pad columns if needed
        if current_y < target_y:
            num_new_cols = target_y - current_y
            # Generate random values for new columns
            new_cols = np.random.normal(loc=0.01, scale=.2, size=(matrix.shape[0], num_new_cols))
            # Create new column names to avoid collision
            new_col_names = []
            base_name = "pad_col"
            i = 0
            while len(new_col_names) < num_new_cols:
                candidate = f"{base_name}_{i}"
                if candidate not in matrix.columns:
                    new_col_names.append(candidate)
                i += 1
            new_cols_df = pd.DataFrame(new_cols, columns=new_col_names)
            matrix = pd.concat([matrix, new_cols_df], axis=1)

        # If matrix is larger than target, just truncate
        matrix = matrix.iloc[:target_x, :target_y]
        return matrix
    def get_embedding_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the loaded embeddings.
        
        Returns:
            Dictionary containing embedding statistics
        """
        if self.embeddings_df is None:
            return {"error": "No embeddings loaded"}
        
        stats = {
            "total_proteins": len(self.embeddings_df.columns),
            "embedding_dimensions": len(self.embeddings_df.index),
            "shape": self.embeddings_df.shape,
            "memory_usage_mb": self.embeddings_df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        if self.filtered_embeddings is not None:
            stats.update({
                "filtered_proteins": len(self.filtered_embeddings.columns),
                "filtered_shape": self.filtered_embeddings.shape,
                "proteins_found": len(set(self.found_proteins)),
                "proteins_missing": len(set(self.missing_proteins)),
                "filtered_memory_usage_mb": self.filtered_embeddings.memory_usage(deep=True).sum() / 1024 / 1024
            })
        
        return stats
    
    def save_filtered_embeddings(self, output_path: str, include_index: bool = False) -> None:
        """
        Save filtered embeddings to a CSV file.
        
        Args:
            output_path: Path to save the filtered embeddings
            include_index: Whether to include row index in saved file
        """
        if self.filtered_embeddings is None:
            raise ValueError("No filtered embeddings available. Call filter_embeddings_for_proteins() first.")
        
        try:
            self.filtered_embeddings.to_csv(output_path, index=include_index)
            print(f"Filtered embeddings saved to: {output_path}")
            print(f"Saved shape: {self.filtered_embeddings.shape}")
            
        except Exception as e:
            raise Exception(f"Error saving filtered embeddings to {output_path}: {str(e)}")
  
    def process_embeddings_pipeline(self, protein_file_path: str, 
                                  output_path: str,
                                  protein_column: int = 1,
                                  header: Optional[int] = None,
                                  low_memory: bool = False,
                                  fill_missing_with_zeros: bool = True) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Complete pipeline to process embeddings: load -> filter -> save.
        
        Args:
            protein_file_path: Path to file containing proteins of interest
            output_path: Path to save filtered embeddings
            protein_column: Column index for protein IDs (0-based)
            header: Header row index for protein file
            low_memory: Use low_memory option for loading
            fill_missing_with_zeros: Fill missing proteins with zeros
            
        Returns:
            Tuple of (filtered_embeddings, statistics)
        """
        print("Starting embedding processing pipeline...")
        
        # Step 1: Load embeddings
        print("\n1. Loading embeddings...")
        self.load_embeddings(low_memory=low_memory)
        
        # Step 2: Load proteins of interest
        print("\n2. Loading proteins of interest...")
        proteins = self.load_protein_list(protein_file_path, protein_column, header)
        
        # Step 3: Filter embeddings
        print("\n3. Filtering embeddings...")
        filtered = self.filter_embeddings_for_proteins(proteins, fill_missing_with_zeros)
        
        # Step 4: Save results
        print("\n4. Saving filtered embeddings...")
        self.save_filtered_embeddings(output_path)
        
        # Step 5: Get statistics
        stats = self.get_embedding_statistics()
        
        print("\n5. Pipeline completed successfully!")
        print(f"Final statistics: {stats}")
        
        return filtered, stats