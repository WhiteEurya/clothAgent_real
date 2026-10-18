"""Bundle a frozen Arm6 graph and its evidence for replay from repository root."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments.graph_policy import load_frozen, frozen, checked_file
from cloth_agent.experiments.discovery_data import file_hash
from cloth_agent.experiments.arm6_traj_replay import compile_templates


def package(source, output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    output.relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    assets = output / 'assets'
    assets.mkdir()
    graph = load_frozen(source)
    trajectory = load_frozen(checked_file(graph['source_trajectory']))

    def bundle(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and value.get('sha256'):
                src = checked_file(value)
                target = assets / (value['sha256'] + '_' + src.name)
                if not target.exists():
                    shutil.copyfile(src, target)
                result = copy.deepcopy(value)
                result['path'] = str(target.relative_to(root))
                return result
            return {key: bundle(item) for key, item in value.items()}
        if isinstance(value, list):
            return [bundle(item) for item in value]
        return value

    trajectory = bundle(trajectory)
    graph = bundle(graph)
    frozen(output / 'trajectory.json', trajectory)
    graph['source_trajectory'] = dict(path=str((output / 'trajectory.json').relative_to(root)),
                                    sha256=file_hash(output / 'trajectory.json'))
    frozen(output / 'graph.json', graph)
    _, _, templates = compile_templates(output / 'graph.json')
    print(json.dumps(dict(graph=str(output / 'graph.json'), edges=list(templates))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.graph, args.output)
