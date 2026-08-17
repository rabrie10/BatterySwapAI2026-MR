
print('import start')

import os
import pickle
from pathlib import Path

# These utilities must be part of submission repo
from batteryswap_public.utils import make_submissions
from batteryswap_public.interfaces import Planner

# ensure classes that might be referenced in pickle is imported
from batteryswap_example.train import * 


def pickle_loader(path : str):
    
    def load() -> Planner:
        with open(path, "rb") as f:
            return pickle.load(f)
    return load

def main():
    print('main start')

    # Load a trained model from repo
    default_planner_path = 'batteryswap_example/planners/best.pickle'
    planner_path = Path(os.environ.get('BATTERYSWAP_PLANNER_PATH', default_planner_path))
    loader = pickle_loader(planner_path)

    # NOTE: On HuggingFace the dataset will be in /tmp/data
    dataset_path = Path(os.environ.get('BATTERYSWAP_DATASET_PATH', '/tmp/data'))

    splits = os.environ.get('BATTERYSWAP_SPLITS', 'public,private').split(',')
    make_submissions(loader, dataset_path=dataset_path, splits=splits)

    # NOTE: On HuggingFace the output must be submission.csv
    assert os.path.exists('submission.csv')

if __name__ == '__main__':
    main()

