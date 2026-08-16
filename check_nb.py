import json
import ast

try:
    with open('01_explore_data.ipynb') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Failed to read notebook: {e}")
    exit(1)

print("Checking for execution errors in outputs...")
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        # Execution errors
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                print(f"\n[Execution Error in Cell {i}]")
                print(f"Name: {output.get('ename')}")
                print(f"Value: {output.get('evalue')}")
                print("Traceback:")
                for line in output.get('traceback', []):
                    print(line)
        
        # Syntax errors
        source = "".join(cell.get('source', []))
        # Filter out magics (lines starting with % or !)
        source_lines = []
        for line in source.split('\n'):
            if line.strip().startswith('%') or line.strip().startswith('!'):
                source_lines.append('')
            else:
                source_lines.append(line)
        clean_source = '\n'.join(source_lines)
        
        try:
            ast.parse(clean_source)
        except SyntaxError as e:
            print(f"\n[Syntax Error in Cell {i}]")
            print(f"Line {e.lineno}: {e.msg}")
            print(e.text)
