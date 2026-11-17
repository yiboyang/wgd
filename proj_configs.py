from pathlib import Path

project_dir = Path(__file__).resolve().parent
slurm_jobs_dir = project_dir / "slurm_jobs"

fixed_size_tfds_datasets = [
    'mnist', 'cifar10', 'cifar100'
]

# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
biggan_class_names_to_ids = {
    'basenji': 253,
    'beagle': 162,
    'church': 497,
    'tabby': 281,
    'jay': 17,
    'magpie': 18,
}

# Conventions for names/tags of dirs/groups used in training.
TRAIN_COLLECTION = "train"
VAL_COLLECTION = "val"
CHECKPOINTS_DIR_NAME = "checkpoints"

