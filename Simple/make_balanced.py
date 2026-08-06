import os
import random
import shutil
from pathlib import Path


def balance_dataset_folder(input_folder: str | Path):
    source_dir = Path(input_folder).resolve()
    target_dir = source_dir.parent / f"{source_dir.name}_balanced"

    dir_real = source_dir / "0Real"
    dir_fake = source_dir / "1Fake"

    files_real = [f for f in dir_real.iterdir()]
    files_fake = [f for f in dir_fake.iterdir()]

    count_real = len(files_real)
    count_fake = len(files_fake)

    print(f"Found videos in source:")
    print(f"   • 0Real: {count_real}")
    print(f"   • 1Fake: {count_fake}")

    target_real = target_dir / "0Real"
    target_fake = target_dir / "1Fake"

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_real.mkdir(parents=True, exist_ok=True)
    target_fake.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {count_real} videos from 0Real...")
    for file in files_real:
        shutil.copy2(file, target_real / file.name)

    sample_size = min(count_real, count_fake)
    selected_fakes = random.sample(files_fake, sample_size)

    print(f"Random sample {sample_size} from  1Fake."    )
    for file in selected_fakes:
        shutil.copy2(file, target_fake / file.name)

    print(f"\nBakanced dataset creation complete")
    print(f" {target_dir}")
    print(f"   0Real: {len(list(target_real.iterdir()))}")
    print(f"   Fake: {len(list(target_fake.iterdir()))}")


from collections import defaultdict


def balance_dataset_folder2(input_folder: str | Path):
    "Not random sample, but sampling quota per method"
    source_dir = Path(input_folder).resolve()
    target_dir = source_dir.parent / f"{source_dir.name}_balanced"

    dir_real = source_dir / "0Real"
    dir_fake = source_dir / "1Fake"

    files_real = [f for f in dir_real.iterdir() if f.is_file()]
    files_fake = [f for f in dir_fake.iterdir() if f.is_file()]

    count_real = len(files_real)
    count_fake = len(files_fake)

    print(f"Found videos in source:")
    print(f"   • 0Real: {count_real}")
    print(f"   • 1Fake: {count_fake}")

    fake_by_method = defaultdict(list)
    for f in files_fake:
        method = f.name.split("_")[0]
        fake_by_method[method].append(f)

    num_methods = len(fake_by_method)
    if num_methods == 0:
        print("No methods, empty folder")
        return

    quota_per_method = count_real // num_methods
    remainder = count_real % num_methods

    print(f"\nN methods:: {num_methods}")
    print(f"   Quota per method: ~{quota_per_method}")

    selected_fakes = []

    for i, (method, files) in enumerate(fake_by_method.items()):
        target_count = quota_per_method + (1 if i < remainder else 0)

        actual_sample_size = min(target_count, len(files))

        method_samples = random.sample(files, actual_sample_size)
        selected_fakes.extend(method_samples)

        print(
            f"   Method '{method}': sampled {len(method_samples)} from {len(files)} videos"
        )

    target_real = target_dir / "0Real"
    target_fake = target_dir / "1Fake"

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_real.mkdir(parents=True, exist_ok=True)
    target_fake.mkdir(parents=True, exist_ok=True)


    print(f"\nCopying {count_real} videos from 0Real...")
    for file in files_real:
        shutil.copy2(file, target_real / file.name)

    print(f"Sampling {len(selected_fakes)} videos from 1Fake...")
    for file in selected_fakes:
        shutil.copy2(file, target_fake / file.name)

    print(f"\nBakanced dataset creation complete")
    print(f" {target_dir}")
    print(f"   0Real: {len(list(target_real.iterdir()))}")
    print(f"   Fake: {len(list(target_fake.iterdir()))}")


if __name__ == "__main__":
    ROOT_PATH = Path("./datasets/CELEB_extract_1v0/")
    # for fn in ["training", "validation", "test", "eval"]:
    for fn in ["training", "validation", "test"]:
        balance_dataset_folder2(ROOT_PATH / fn)