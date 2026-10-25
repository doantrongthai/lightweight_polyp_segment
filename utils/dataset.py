# utils/dataset.py
import os
import zipfile

_DRIVE_IDS = {
    "TrainDataset": "1kuCLB6FTvQdHbbMeTZDY22gNLVl-n-xZ",
    "TestDataset":  "1PlEo97uOUx68m9G_2VENeVDkzRuioYJ3",
}

# ── Kiểm tra dataset ────────────────────────────────────────────────────────

def _is_train_ready(root: str) -> bool:
    """
    Sau khi giải nén, train nằm thẳng tại:
      root/image/   (ảnh gốc)
      root/masks/   (ground truth)
    """
    img_dir  = os.path.join(root, "image")
    mask_dir = os.path.join(root, "masks")
    return (
        os.path.isdir(img_dir)  and len(os.listdir(img_dir))  > 0 and
        os.path.isdir(mask_dir) and len(os.listdir(mask_dir)) > 0
    )

def _is_test_ready(root: str) -> bool:
    """
    Sau khi giải nén, các bộ test nằm thẳng tại:
      root/CVC-300/
      root/CVC-ClinicDB/
      ...
    Mỗi folder có subfolder images/ và masks/.
    """
    expected = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB",
                "ETIS-LaribPolypDB", "Kvasir"]
    return all(
        os.path.isdir(os.path.join(root, d, "images")) and
        len(os.listdir(os.path.join(root, d, "images"))) > 0
        for d in expected
    )

# ── Download + giải nén ────────────────────────────────────────────────────

def _download_and_extract(file_id: str, name: str, root: str) -> None:
    try:
        import gdown
    except ImportError:
        raise ImportError("Thiếu thư viện gdown. Cài bằng: pip install gdown")

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, f"{name}.zip")

    print(f"[dataset] Đang tải {name} từ Google Drive ...")
    gdown.download(id=file_id, output=zip_path, quiet=False)

    print(f"[dataset] Đang giải nén {name}.zip → {root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)          # giải nén thẳng vào root, không tạo subfolder thêm
    os.remove(zip_path)
    print(f"[dataset] {name} sẵn sàng tại: {root}")

# ── API chính ──────────────────────────────────────────────────────────────

def prepare_datasets(
    root: str = "./dataset",
    need_train: bool = True,
    need_test:  bool = True,
) -> None:
    """
    Cấu trúc thực tế sau khi giải nén:
      dataset/
        image/              ← train images
        masks/              ← train masks
        CVC-300/images/     ┐
        CVC-ClinicDB/...    │ test sets
        CVC-ColonDB/...     │
        ETIS-LaribPolypDB/  │
        Kvasir/             ┘
    """
    if need_train:
        if not _is_train_ready(root):
            print(f"[dataset] Không tìm thấy train data tại '{root}/image'.")
            _download_and_extract(_DRIVE_IDS["TrainDataset"], "TrainDataset", root)
        else:
            print(f"[dataset] Train data đã có tại '{root}/image' & '{root}/masks'.")

    if need_test:
        if not _is_test_ready(root):
            print(f"[dataset] Không tìm thấy test data tại '{root}'.")
            _download_and_extract(_DRIVE_IDS["TestDataset"], "TestDataset", root)
        else:
            print(f"[dataset] Test data đã có tại '{root}'.")