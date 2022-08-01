names = []
numeric = []
boolean = []

with open(names_file_path) as names_f:
    for ln in names_f.readlines():
        if ':' in ln:
            col_name = ln.split(':', 1)[0]
            names.append(col_name)
            if 'continuous' in ln:
                numeric.append(col_name)
            elif 'f, t' in ln:
                boolean.append(col_name)
    names.append('diagnosis')
