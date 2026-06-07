#!/bin/bash
# ============================================
# Script cài đặt môi trường Vehicle Detection
# Chạy: bash setup.sh
# ============================================

echo "=========================================="
echo "  Vehicle Detection - Cài đặt môi trường"
echo "=========================================="

# Kiểm tra Python
python_version=$(python3 --version 2>&1)
echo "✅ Python: $python_version"

# Tạo virtual environment
echo ""
echo "📦 Tạo virtual environment..."
python3 -m venv venv

# Kích hoạt venv
echo "✅ Kích hoạt virtual environment..."
source venv/bin/activate

# Nâng cấp pip
echo ""
echo "⬆️  Nâng cấp pip..."
pip install --upgrade pip

# Cài đặt PyTorch (CPU version mặc định)
echo ""
echo "🔥 Cài đặt PyTorch..."
echo "   Nếu bạn có GPU NVIDIA, hãy truy cập https://pytorch.org để lấy lệnh cài đặt phù hợp."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện còn lại
echo ""
echo "📚 Cài đặt thư viện..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Cài đặt hoàn tất!"
echo ""
echo "Bước tiếp theo:"
echo "  1. Tải Dataset và đặt vào thư mục Dataset/"
echo "  2. Tải model_vehicle.pth và đặt vào thư mục gốc"
echo "  3. Huấn luyện: python train.py"
echo "  4. Kiểm tra:   python test.py"
echo "  5. Giao diện:  python GiaoDien.py"
echo "=========================================="
