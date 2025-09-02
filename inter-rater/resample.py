from sklearn.utils import resample

# giả sử bạn có list 500 ảnh id
image_ids = list(range(500))

# Resample 500 ảnh có hoàn lại
sample = resample(image_ids, replace=True, n_samples=len(image_ids), random_state=42)

print(len(sample))       # vẫn là 500
print(sample[:10])       # xem thử 10 ảnh đầu, có thể trùng lặp