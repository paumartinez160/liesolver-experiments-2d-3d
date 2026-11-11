"""Run the four main benchmarks sequentially and compare each pair."""
import run
from pipeline.comparison import compare

if __name__ == '__main__':
    for dimension in (2, 3):
        config = run.ROOT / f'configs_{dimension}d/wave_{dimension}d_sine_4pi.yaml'
        folders = [run.run(config, method) for method in ('liesolver', 'classical')]
        print(compare(folders, run.ROOT / 'results'))
