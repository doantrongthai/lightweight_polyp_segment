# Polyp Segmentation — Hướng dẫn chạy

## Yêu cầu

```bash
pip install -r requirements.txt
```

---

## Cấu trúc thư mục

```
project/
├── main.py
├── models/
├── losses/
├── utils/
└── dataset/
    ├── TrainDataset/
    │   ├── images/
    │   └── masks/
    └── TestDataset/
        ├── CVC-300/
        ├── CVC-ClinicDB/
        ├── CVC-ColonDB/
        ├── ETIS-LaribPolypDB/
        └── Kvasir/
```

Dataset sẽ **tự động tải về** lần đầu nếu chưa có. Cần kết nối internet.

---

## Train + Test (1 seed — seed mặc định: 42)

Chạy toàn bộ: train xong tự động test trên cả 5 bộ.

```bash
python main.py --model <tên_model>
```

**Ví dụ:**

```bash
python main.py --model polyp_pvt
```

Kết thúc sẽ in bảng kết quả đầy đủ (Dice, IoU, NSD, HD95, ...) cho cả 5 dataset.

---

## Train + Test (5 seed — để báo cáo mean ± std)

Chạy lần lượt 5 seed: `42, 123, 456, 789, 2024`.

```bash
python main.py --model <tên_model> --multiseed
```

**Ví dụ:**

```bash
python main.py --model polyp_pvt --multiseed
```

Chạy lần lượt 5 seed: `42, 123, 456, 789, 2024`.  
Kết thúc in bảng tổng hợp mean ± std và xuất LaTeX table sẵn.

---

## Chỉ test (đã có weights)

Test 1 seed trên cả 5 dataset:

```bash
python main.py --model <tên_model> --test_only --weights <đường_dẫn_weights>
```

**Ví dụ — test weights của seed 42:**

```bash
python main.py --model polyp_pvt --test_only --weights ./checkpoints/polyp_pvt/best_seed42.pth
```

---

## Resume nếu train bị ngắt

```bash
python main.py --model <tên_model> --resume
```

Tự động tiếp tục từ epoch cuối đã lưu. Nếu không tìm thấy checkpoint thì train lại từ đầu.

---

## Kết quả lưu ở đâu

| File | Nội dung |
|---|---|
| `checkpoints/<model>/best_seed42.pth` | Weights tốt nhất (1 seed) |
| `checkpoints/<model>/best_seed<N>.pth` | Weights tốt nhất của seed N (multiseed) |
| `checkpoints/<model>/train_seed<N>.log` | Log train theo epoch |
| `checkpoints/<model>/history_seed<N>.json` | Dice theo epoch từng dataset |
| `checkpoints/<model>/<model>_multiseed.json` | Kết quả tổng hợp 5 seed |

---

## 5 dataset test

| Dataset | Số ảnh |
|---|---|
| Kvasir | 100 |
| CVC-ClinicDB | 62 |
| CVC-ColonDB | 380 |
| CVC-300 | 60 |
| ETIS-LaribPolypDB | 196 |

Train trên `TrainDataset`, test trên cả 5 bộ trên — không overlap.