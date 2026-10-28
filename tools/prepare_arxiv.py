#!/usr/bin/env python3
"""Package the manuscript as one self-contained TeX file and three figures."""
import argparse
from pathlib import Path
import re
import shutil
import zipfile


def prepare(output):
    source = Path(__file__).resolve().parents[1] / 'paper'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    text = (source / 'main.tex').read_text()
    for name in ('generated.tex', 'secondary_generated.tex', 'expanded_generated.tex'):
        definitions = (source / name).read_text()
        definitions = '\n'.join(line for line in definitions.splitlines()
                                if not line.startswith('% Generated')) + '\n'
        command = '\\input{' + name + '}'
        if text.count(command) != 1:
            raise ValueError(f'Expected exactly one input for {name}')
        text = text.replace(command, definitions)
    if re.search(r'\\(?:input|include)\s*\{', text):
        raise ValueError('Unexpected remaining source dependency')
    (output / 'main.tex').write_text(text)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    names = ('performance-reduction.png', 'paired-effects.png', 'label-noise.png')
    for name in names:
        shutil.copyfile(source / 'figures' / name, figures / name)
    archive = output.parent / 'BUSBRA_arxiv_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [output / 'main.tex', *(figures / name for name in names)]:
            bundle.write(path, path.relative_to(output))
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Directory for main.tex and figures; ZIP is written beside it')
    print(prepare(parser.parse_args().output))
