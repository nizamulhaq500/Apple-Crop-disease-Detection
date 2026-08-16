import json

with open('01_explore_data.ipynb') as f:
    nb = json.load(f)

for i in range(6):
    cell = nb.get('cells', [])[i]
    if cell.get('cell_type') == 'code':
        print(f"\n--- Code Cell {i} ---")
        print("".join(cell.get('source', [])))
