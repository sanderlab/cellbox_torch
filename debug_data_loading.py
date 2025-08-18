#!/usr/bin/env python3

import pickle
import json
import sys
import os
from pathlib import Path

# Add the project root to sys.path to import filtering functions
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Abdullah_kuziez.preprocessing.pre_processing_py_fxns.filtering_functions import load_data

def debug_load_data():
    """Debug function to check what's in the data_6hr dictionary"""
    
    intermediate_dir_6hr = Path("Abdullah_kuziez/preprocessing/TNBC_notebooks/intermediate_files_TNBC/6hr")
    
    print("Loading 6hr data...")
    try:
        data_6hr = load_data(intermediate_dir_6hr, "6hr")
        print("✓ Data loaded successfully")
        
        print("\nKeys in data_6hr:")
        for key in data_6hr.keys():
            print(f"  - {key}: {type(data_6hr[key])}")
            
        # Check specific keys that are causing issues
        problematic_keys = ['cell_lines', 'control_data_by_cell_line', 'control_data_by_cell_line_coeffvar']
        
        for key in problematic_keys:
            if key in data_6hr:
                value = data_6hr[key]
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                if isinstance(value, str):
                    print(f"  Value (first 200 chars): {value[:200]}")
                    print("  ❌ This should not be a string!")
                else:
                    print(f"  ✓ Correct type: {type(value)}")
                    if hasattr(value, 'keys'):
                        print(f"  Keys: {list(value.keys())}")
                    elif hasattr(value, '__len__'):
                        print(f"  Length: {len(value)}")
            else:
                print(f"\n❌ Key '{key}' not found in data_6hr")
                
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_load_data()
