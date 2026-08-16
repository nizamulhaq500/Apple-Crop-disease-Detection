import json

with open('01_explore_data.ipynb') as f:
    nb = json.load(f)

cells = nb['cells']

# Find indices based on content to be safe
split_code_indices = []
count_split_indices = []

for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'split_ratios =' in source and 'train_end =' in source:
            split_code_indices.append(i)
        if 'for split_name in ["train", "val", "test"]:' in source:
            count_split_indices.append(i)

print("Split code cells:", split_code_indices)
print("Count split cells:", count_split_indices)

# We want the split code to be BEFORE the count split code.
# The current order is Count (Cell 3), Split (Cell 4), Split (Cell 5).
# So we keep Split, then Count, and remove the duplicate Split.

# Let's just build a new list of cells
new_cells = []
for i, cell in enumerate(cells):
    if i in split_code_indices[1:]:
        # drop duplicates of split code
        continue
    new_cells.append(cell)

cells = new_cells
# recalculate indices
split_code_idx = next(i for i, c in enumerate(cells) if 'split_ratios =' in "".join(c.get('source', [])))
count_split_idx = next(i for i, c in enumerate(cells) if 'for split_name in ["train", "val", "test"]:' in "".join(c.get('source', [])))

if count_split_idx < split_code_idx:
    # Swap them
    cells[count_split_idx], cells[split_code_idx] = cells[split_code_idx], cells[count_split_idx]

# Clear outputs
for cell in cells:
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None

nb['cells'] = cells

with open('01_explore_data.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed successfully!")
