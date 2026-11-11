"""Run one experiment per toy dataset"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path

from drifting.data import NAMES
from drifting.train import Config, train


def main():
    results_folder = Path('output0.05')
    seed = 42
    device = 'cpu'  # Change to 'cuda' to use GPU

    # Check before starting, so existing results are not overwritten.
    for dataset in NAMES:
        output_folder = results_folder / dataset

        if output_folder.exists():
            raise FileExistsError(
                f'{output_folder} already exists. '
                'Choose a different results_folder for this experiment.'
            )

    for dataset in NAMES:
        print(f'\nTraining on {dataset}', flush=True)

        config = Config(
            dataset=dataset,
            seed=seed,
            device=device,
            temperature=0.05
        )

        train(
            config,
            out=results_folder / dataset,
        )

    print('\nAll experiments finished.', flush=True)


if __name__ == '__main__':
    main()