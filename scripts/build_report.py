"""Build evidence-derived LaTeX fragments, then compile only if a TeX engine exists."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
import shutil
import subprocess
from src.utils import ROOT, read_json, environment


def tex_escape(value: str) -> str:
    return ''.join({'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#',
                    '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}'}.get(c, c) for c in str(value))


def generate_fragments() -> None:
    report = ROOT/'report'
    report.mkdir(exist_ok=True)
    rows = []
    for name, filename, split in [('Validation', 'validation_metrics.json', 'validation'),
                                  ('DUTS-TE', 'duts_test_metrics.json', 'duts-test'),
                                  ('Captured photos', 'captured_metrics.json', 'captured')]:
        file = ROOT/'results'/filename
        if not file.exists():
            rows.append(name+r' & \multicolumn{7}{l}{Not evaluated; prerequisite evidence pending} \\')
        else:
            value = read_json(file)
            if value['split'] != split:
                raise ValueError('Report source has incorrect split')
            rows.append(name+' & '+' & '.join(f'{value[k]:.4f}' for k in ('miou', 'bf1', 'q'))+
                        f" & {value['cpu_ms']:.3f} & {value['size_bytes']} & {value['ces']:.4f} & {value['images']} "+r'\\')
    table = '\n'.join([r'\begin{tabular}{lrrrrrrr}', r'\toprule',
        r'Set & mIoU & BF1 & Q & CPU ms & Bytes & CES & $N$ \\', r'\midrule', *rows,
        r'\bottomrule', r'\end{tabular}'])
    (report/'results_table.tex').write_text(table, encoding='utf-8')
    split_file = ROOT/'data/splits/metadata.json'
    split_text = 'Dataset preparation pending; no real manifests are claimed.'
    if split_file.exists():
        split = read_json(split_file)
        split_text = f"Actual saved split: {split['train']} training and {split['validation']} validation images; seed {split['seed']}."
    (report/'environment.tex').write_text(tex_escape(environment()['cpu_model'])+'. '+split_text, encoding='utf-8')
    selection_file = ROOT/'results/selection.json'
    architecture = 'Final architecture selection is pending measured validation comparisons.'
    if selection_file.exists():
        selection = read_json(selection_file)
        from src.inference import Predictor
        predictor = Predictor(selection['weights'])
        count = sum(p.numel() for p in predictor.model.parameters())
        architecture = tex_escape(f"Validation-selected candidate: {selection['name']}. "
            f"Architecture: {predictor.config}; learned parameters: {count}; "
            f"inference resolution: {predictor.resolution}; threshold: {predictor.threshold}.")
    (report/'selected_architecture.tex').write_text(architecture, encoding='utf-8')
    captured_index = ROOT/'results/captured_figures/index.json'
    lines = []
    if captured_index.exists():
        for record in read_json(captured_index):
            # Figure filenames are generated numeric names, not untrusted sample IDs.
            figure = Path(record['figure']).name
            lines += [r'\begin{figure}[htbp]\centering',
                      r'\includegraphics[width=\linewidth]{../results/captured_figures/'+figure+'}',
                      r'\caption{Captured photo '+tex_escape(record['id'])+r': image, masks, and true/predicted overlays.}',
                      r'\end{figure}']
    else:
        lines = ['No genuine captured photographs or annotations have been supplied. Evaluation and mandatory overlays remain incomplete.']
    (report/'captured_figures.tex').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-only', action='store_true')
    p.add_argument('--engine', help='Optional explicit pdflatex or tectonic executable path')
    args = p.parse_args()
    generate_fragments()
    if args.source_only:
        print('Updated report fragments from existing evidence')
        return
    engine = args.engine or shutil.which('pdflatex') or shutil.which('tectonic')
    if not engine:
        print('LaTeX source prepared; no compiler is available. PDF compilation and visual review remain pending.')
        return
    if 'tectonic' in Path(engine).name.lower():
        subprocess.run([engine, '--keep-logs', 'main.tex'], cwd=ROOT/'report', check=True)
        print(ROOT/'report/main.pdf')
        return
    for _ in range(2):
        subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=ROOT/'report', check=True)
    print(ROOT/'report/main.pdf')


if __name__ == '__main__':
    main()
