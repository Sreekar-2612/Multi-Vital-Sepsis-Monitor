import nbformat

notebook_path = 'Realtime_Sepsis_Detection_Architecture.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if "df.ffill().fillna(method='bfill')" in cell.source:
            cell.source = cell.source.replace("df.ffill().fillna(method='bfill')", "df.ffill().bfill()")

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook fixed successfully!")
