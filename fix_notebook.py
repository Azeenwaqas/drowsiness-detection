import json
import sys

def fix_notebook(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('Output_Images', exist_ok=True)\n",
                "os.makedirs('Dataset', exist_ok=True)"
            ]
        }
        
        # Insert as first cell
        nb['cells'].insert(0, new_cell)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Successfully updated Notebook.ipynb")
    except Exception as e:
        print(f"Error updating Notebook.ipynb: {e}")

if __name__ == "__main__":
    fix_notebook(sys.argv[1])
