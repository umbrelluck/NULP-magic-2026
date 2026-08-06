import cv2
from pathlib import Path
import urllib
import os

YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"


def get_yunet_detector():
    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"Downloading YuNet ({YUNET_MODEL_PATH})...")
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
        print("Done")

    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(320, 320),
        score_threshold=0.6, 
        nms_threshold=0.3,
    )
    return detector




def crop_faces_celebdf(
    dataset_root: Path | str,
    output_dir: Path | str = "./cropped_celebdf",
    frames_per_video: int = 3,
    test_limit_id: int = 50, 
    target_size: tuple[int, int] = (299, 299),
    margin_ratio: float = 0.3,
):
    dataset_root = Path(dataset_root)
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    folders_map = {
        "Celeb-real": "0Real",
        "YouTube-real": "0Real",
        "Celeb-synthesis": "1Fake"
    }

    detector = get_yunet_detector()
    saved_count = {"0Real": 0, "1Fake": 0}

    for folder_name, category_name in folders_map.items():
        folder_path = dataset_root / folder_name
        if not folder_path.exists():
            print(f"Not found {folder_path}, skip")
            continue

        videos = [f for f in folder_path.iterdir() if f.suffix.lower() == ".mp4"]
        print(f"Working on {folder_name}: {len(videos)} videos...")

        for vid_path in videos:
            filename = vid_path.stem
            
            if filename.startswith("id"):
                try:
                    actor_id = int(filename.split("_")[0].replace("id", ""))
                except ValueError:
                    actor_id = -1
            else:
                actor_id = 999 

            if actor_id < test_limit_id:
                continue

            cap = cv2.VideoCapture(str(vid_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 10:
                cap.release()
                continue

            frame_indices = [
                int(total_frames * (i + 1) / (frames_per_video + 1))
                for i in range(frames_per_video)
            ]

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if not success or frame is None:
                    continue

                h_img, w_img, _ = frame.shape
                detector.setInputSize((w_img, h_img))
                _, faces = detector.detect(frame)

                if faces is not None and len(faces) > 0:
                    x, y, w, h = map(int, faces[0][:4])
                    margin_x = int(w * margin_ratio)
                    margin_y = int(h * margin_ratio)

                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(w_img, x + w + margin_x)
                    y2 = min(h_img, y + h + margin_y)

                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    face_img = cv2.resize(
                        face_img, target_size, interpolation=cv2.INTER_CUBIC
                    )

                    output_filename = f"{folder_name}_{vid_path.stem}_f{frame_idx}.jpg"
                    save_path = dest_dir / category_name / output_filename
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(save_path), face_img)
                    saved_count[category_name] += 1

            cap.release()

    print(f"Extracted:")
    print(f"   0Real: {saved_count['0Real']} frames")
    print(f"   1Fake: {saved_count['1Fake']} frames")


import random
import shutil

def build_balanced_eval_celebdf(
    cropped_dir: Path | str,
    output_eval_dir: Path | str = "./datasets/CelebDF_eval",
    images_per_class: int = 150,
):
    """ 
    Making just eval subset, to be forwarded for FF++ Deepfake testing
    """

    cropped_path = Path(cropped_dir)
    target_dir = Path(output_eval_dir)

    dir_real = cropped_path / "0Real"
    dir_fake = cropped_path / "1Fake"

    files_real = [f for f in dir_real.iterdir() if f.is_file()] if dir_real.exists() else []
    files_fake = [f for f in dir_fake.iterdir() if f.is_file()] if dir_fake.exists() else []

    print(f"Found images:")
    print(f"   • 0Real: {len(files_real)}")
    print(f"   • 1Fake: {len(files_fake)}")

    if len(files_real) < images_per_class or len(files_fake) < images_per_class:
        print(f"!!!! Less than {images_per_class} images per class.")

    sample_real_cnt = min(images_per_class, len(files_real))
    sample_fake_cnt = min(images_per_class, len(files_fake))

    selected_real = random.sample(files_real, sample_real_cnt)
    selected_fake = random.sample(files_fake, sample_fake_cnt)

    target_real = target_dir / "0Real"
    target_fake = target_dir / "1Fake"

    if target_dir.exists():
        shutil.rmtree(target_dir)

    target_real.mkdir(parents=True, exist_ok=True)
    target_fake.mkdir(parents=True, exist_ok=True)

    print(f"\Copying {images_per_class} videos to balanced eval...")
    for file in selected_real:
        shutil.copy2(file, target_real / file.name)

    for file in selected_fake:
        shutil.copy2(file, target_fake / file.name)

    print(f"\nBalanced eval completed:")
    print(f" {target_dir}")
    print(f"   0Real: {len(list(target_real.iterdir()))} шт.")
    print(f"   1Fake: {len(list(target_fake.iterdir()))} шт.")


# ===================================================
def crop_faces_celebdf_split(
    dataset_root: Path | str,
    output_dir: Path | str = "./datasets/CELEB_extract_1v0",
    frames_per_video: int = 3,
    target_size: tuple[int, int] = (299, 299),
    margin_ratio: float = 0.3,
    split_ratios: tuple[float, float, float] = (0.80, 0.15, 0.05),  # train/val/test
    seed: int = 42,
):
    """ 
    Making all subsets, except eval
    """
    dataset_root = Path(dataset_root)
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    train_ratio, val_ratio, test_ratio = split_ratios
    assert (
        abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5
    ), "Sum of ratio != 1.0"

    folders_map = {
        "Celeb-real": "0Real",
        "Celeb-synthesis": "1Fake",
    }

    actor_ids = set()
    for folder in folders_map.keys():
        folder_path = dataset_root / folder
        if folder_path.exists():
            for f in folder_path.glob("*.mp4"):
                if f.stem.startswith("id"):
                    try:
                        actor_id = int(f.stem.split("_")[0].replace("id", ""))
                        actor_ids.add(actor_id)
                    except ValueError:
                        pass

    sorted_actors = sorted(list(actor_ids))
    random.seed(seed)
    random.shuffle(sorted_actors)

    n_actors = len(sorted_actors)
    n_val = max(1, int(n_actors * val_ratio)) if n_actors >= 3 else 0
    n_test = max(1, int(n_actors * test_ratio)) if n_actors >= 3 else 0
    n_train = n_actors - n_val - n_test

    train_actors = set(sorted_actors[:n_train])
    val_actors = set(sorted_actors[n_train : n_train + n_val])
    test_actors = set(sorted_actors[n_train + n_val :])

    print(f"Actors (total {n_actors}):")
    print(f"   Train: {len(train_actors)}")
    print(f"   Val:   {len(val_actors)}")
    print(f"   Test:  {len(test_actors)}")

    detector = get_yunet_detector()

    saved_stats = {
        "train": {"0Real": 0, "1Fake": 0},
        "val": {"0Real": 0, "1Fake": 0},
        "test": {"0Real": 0, "1Fake": 0},
    }

    def get_split_name(vid_path: Path) -> str | None:
        filename = vid_path.stem
        if filename.startswith("id"):
            try:
                actor_id = int(filename.split("_")[0].replace("id", ""))
                if actor_id in train_actors:
                    return "train"
                elif actor_id in val_actors:
                    return "val"
                elif actor_id in test_actors:
                    return "test"
            except ValueError:
                pass
        return None 

    for folder_name, category_name in folders_map.items():
        folder_path = dataset_root / folder_name
        if not folder_path.exists():
            print(f"!!Method {folder_path} not found.")
            continue

        videos = [f for f in folder_path.iterdir() if f.suffix.lower() == ".mp4"]
        print(f"\nWorking on {folder_name}: {len(videos)} videos...")

        for vid_path in videos:
            split_name = get_split_name(vid_path)
            if not split_name:
                continue

            cap = cv2.VideoCapture(str(vid_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 10:
                cap.release()
                continue

            frame_indices = [
                int(total_frames * (i + 1) / (frames_per_video + 1))
                for i in range(frames_per_video)
            ]

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if not success or frame is None:
                    continue

                h_img, w_img, _ = frame.shape
                detector.setInputSize((w_img, h_img))
                _, faces = detector.detect(frame)

                if faces is not None and len(faces) > 0:
                    x, y, w, h = map(int, faces[0][:4])

                    margin_x = int(w * margin_ratio)
                    margin_y = int(h * margin_ratio)
                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(w_img, x + w + margin_x)
                    y2 = min(h_img, y + h + margin_y)

                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    face_img = cv2.resize(
                        face_img, target_size, interpolation=cv2.INTER_CUBIC
                    )

                    output_filename = f"{folder_name}_{vid_path.stem}_f{frame_idx}.jpg"
                    save_path = dest_dir / split_name / category_name / output_filename
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(save_path), face_img)
                    saved_stats[split_name][category_name] += 1

            cap.release()

    print("\nSampling done (ignored YouTube-real)")
    for split_key in ["train", "val", "test"]:
        r_cnt = saved_stats[split_key]["0Real"]
        f_cnt = saved_stats[split_key]["1Fake"]
        print(f" [{split_key.upper()}] -> 0Real: {r_cnt}. | 1Fake: {f_cnt}. (Total: {r_cnt + f_cnt})")


if __name__ == "__main__":
    # crop_faces_celebdf_split(
    #     dataset_root="./datasets/Celeb",
    #     output_dir="./datasets/CELEB_extract_1v0/eval",
    #     test_limit_id=50 # Актори id50, id51... id61 підуть на eval
    # )

    crop_faces_celebdf_split(
        dataset_root="./datasets/Celeb",
        output_dir="./datasets/CELEB_extract_1v0/",
    )

    # build_balanced_eval_celebdf(
    #     cropped_dir="./datasets/CELEB_extract_1v0/eval",
    #     output_eval_dir="./datasets/CELEB_extract_1v0/eval_balanced",
    #     images_per_class=150
    # )