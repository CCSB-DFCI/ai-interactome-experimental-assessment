from pathlib import Path
import subprocess
import re

import numpy as np
import tqdm


# add caching
verbose = False
in_path = Path('../data/interactome3d_2020_01/')
out_file_path = Path('../data/processed/interface-area_atomic-surface_interactome3d_2020_01.tsv')
if out_file_path.exists():
    with open(out_file_path, 'r') as f:
        interface_area = {l.split()[0]: float(l.strip().split()[1]) for l in f.readlines() if l.strip() != ''}
else:
    interface_area = {}
with open(out_file_path, 'a') as out_file:
    for f_path in tqdm.tqdm(list(in_path.glob('*/*.pdb*.pdb'))):
        if f_path.stem in interface_area:
            continue
        pymol_script_path = '../pymol_scripts/interactome3d_2020_01/{}.pml'.format(f_path.stem)
        with open(pymol_script_path, 'w') as f:
            f.write('cd {}\n'.format(str(f_path.parents[0].absolute())))
            f.write('reinitialize\n')
            f.write('load {}\n'.format(f_path.name))

            #f.write('set dot_solvent, 1\n')  # calculate solvent-accessible surface
            f.write('set dot_density, 4\n')  # sampling density from 1(worst)-4(best)
            f.write('copy protein_x_alone, {}\n'.format(f_path.stem))
            f.write('remove (protein_x_alone and chain B)\n')
            f.write('copy protein_y_alone, {}\n'.format(f_path.stem))
            f.write('remove (protein_y_alone and chain A)\n')
            f.write('get_area protein_x_alone\n')
            f.write('get_area protein_y_alone\n')
            f.write('get_area {}\n'.format(f_path.stem))

        ret = subprocess.run('pymol -c {}'.format(pymol_script_path),
                             capture_output=True,
                             shell=True)
        pymol_output = ret.stdout.decode('ascii')
        if 'Error' in pymol_output:
            print(pymol_output)
            #raise UserWarning('error running pymol')
            interface_area[f_path.stem] = np.nan
            continue
        #title = [l[len('TITLE'):].strip() for l in pymol_output.splitlines() if l.startswith('TITLE')][0]
        pattern = ' cmd\.get_area\: ([0-9]+\.[0-9]+) Angstroms\^2\.'
        area_x, area_y, area_xy = [float(re.match(pattern, l).group(1)) for l in pymol_output.splitlines() if re.match(pattern, l)]
        interface_area[f_path.stem] = ((area_x + area_y) - area_xy) / 2
        out_file.write(f'{f_path.stem}\t{interface_area[f_path.stem]}\n')
        if verbose:
            print(ret.stdout.decode())
            print(ret.stderr.decode())