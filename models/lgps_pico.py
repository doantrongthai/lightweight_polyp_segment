import torch
from models.lgps_pico import LGPSPico

def run_smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Đang chạy Smoke Test trên thiết bị: {device} ===")
    
    # Khởi tạo mô hình
    model = LGPSPico(pretrained=False).to(device)
    total_params = model.parameter_count()
    print(f"Tổng số tham số mô hình: {total_params:,}")
    assert 300_000 <= total_params <= 700_000, f"Tham số {total_params} nằm ngoài khoảng yêu cầu!"
    print("[PASS] Kiểm tra ngân sách tham số thành công.")

    # (a) Train mode + Backward với input 352x352 batch 2
    model.train()
    x_train = torch.randn(2, 3, 352, 352, device=device)
    target = torch.randint(0, 2, (2, 352, 352), dtype=torch.float32, device=device)
    
    outputs = model(x_train, return_attn=True)
    loss = model.compute_loss(outputs, target)
    loss.backward()
    print(f"[PASS] Kiểm tra Train mode + Backward thành công (Loss: {loss.item():.4f})")

    # (b) Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x_train, return_attn=False)
    print(f"[PASS] Kiểm tra Eval mode thành công (Shape output: {out_eval.shape})")

    # (c) forward_with_attention
    with torch.no_grad():
        out_attn = model.forward_with_attention(x_train)
    assert isinstance(out_attn, dict) and "pred" in out_attn
    print(f"[PASS] Kiểm tra forward_with_attention thành công (Keys: {list(out_attn.keys())})")

    # (d) Test các kích thước input khác nhau (352x352 batch 2 và 320x320 batch 1)
    with torch.no_grad():
        x_352 = torch.randn(2, 3, 352, 352, device=device)
        res_352 = model(x_352)
        assert res_352.shape == (2, 1, 352, 352)

        x_320 = torch.randn(1, 3, 320, 320, device=device)
        res_320 = model(x_320)
        assert res_320.shape == (1, 1, 320, 320)
    print("[PASS] Kiểm tra đa dạng kích thước input (352x352 và 320x320) thành công.")

    print("=== TẤT CẢ CÁC KIỂM TRA SMOKE TEST ĐÃ ĐẠT (PASS) ===")

if __name__ == "__main__":
    run_smoke_test()
