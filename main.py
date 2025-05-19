# main.py
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import sys
import os

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

# 輸入圖片檔名
img_path = input("請輸入圖片檔案名稱（含副檔名）: ")
if not os.path.exists(img_path):
    print("找不到圖片檔案")
    sys.exit(1)

# 載入模型
model = EfficientNetB7(weights='imagenet', include_top=True)

# 圖像處理
orig_img = Image.open(img_path).convert('RGB')
resized_img = orig_img.resize((600, 600))
img_array = np.clip(np.array(resized_img).astype(np.float32), 0, 255)

# 顯示預覽圖
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(orig_img)
plt.title("原始圖片")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_array.astype(np.uint8))
plt.title("調整後圖片 (600x600)")
plt.axis('off')
plt.suptitle("自駕場景圖像預覽", fontsize=16)
plt.tight_layout()
plt.savefig("圖片預覽.jpg")

# 預測原始圖片
input_image = tf.convert_to_tensor(img_array)
input_image = tf.expand_dims(input_image, axis=0)
adv_image = tf.Variable(input_image)

preds_orig = model(preprocess_input(input_image))
decoded_orig = decode_predictions(preds_orig.numpy(), top=3)[0]
orig_label = tf.argmax(preds_orig[0])
print("\n 原始圖片 top-3 預測結果：")
for rank, (code, label, prob) in enumerate(decoded_orig, 1):
    print(f"{rank}. {label} ({prob:.4f})")
print(f"原始預測類別索引：{orig_label.numpy()}")

# 對抗樣本攻擊（FGSM）
epsilon = 10.0
num_iter = 10
alpha = epsilon / num_iter
loss_history, gradient_images = [], []

for i in range(num_iter):
    with tf.GradientTape() as tape:
        tape.watch(adv_image)
        preds = model(preprocess_input(adv_image))
        loss = tf.keras.losses.sparse_categorical_crossentropy(tf.convert_to_tensor([orig_label]), preds)
    grad = tape.gradient(loss, adv_image)
    signed_grad = tf.sign(grad)
    adv_image.assign_add(alpha * signed_grad)
    adv_image.assign(tf.clip_by_value(adv_image, input_image - epsilon, input_image + epsilon))
    adv_image.assign(tf.clip_by_value(adv_image, 0.0, 255.0))
    loss_history.append(loss.numpy().mean())
    grad_image = tf.reduce_mean(tf.abs(grad[0]), axis=-1).numpy()
    gradient_images.append(grad_image)

# 輸出損失變化圖
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_iter + 1), loss_history, marker='o')
plt.title("對抗樣本攻擊過程中損失變化", fontsize=14)
plt.xlabel("迭代次數")
plt.ylabel("損失值")
plt.grid(True)
plt.tight_layout()
plt.savefig("損失變化圖.jpg")

# 輸出梯度圖
indices = [0, num_iter//3, 2*num_iter//3, num_iter-1]
plt.figure(figsize=(16, 4))
for idx, iter_idx in enumerate(indices):
    plt.subplot(1, len(indices), idx + 1)
    plt.imshow(gradient_images[iter_idx], cmap='viridis')
    plt.title(f"第 {iter_idx + 1} 次擾動")
    plt.axis('off')
plt.suptitle("對抗樣本梯度熱力圖", fontsize=16)
plt.tight_layout()
plt.savefig("梯度熱力圖.jpg")

# 攻擊結果分析
adv_image_np = adv_image.numpy()[0].astype(np.uint8)
perturbation = adv_image_np.astype(np.float32) - img_array.astype(np.float32)
avg_diff = np.mean(np.abs(perturbation))
print(f"\n平均每像素擾動量：{avg_diff:.4f}")

final_preds = model(preprocess_input(tf.expand_dims(adv_image_np.astype(np.float32), axis=0)))
decoded = decode_predictions(final_preds.numpy(), top=3)[0]

print("\n 對抗樣本攻擊後 top-3 預測結果：")
for rank, (code, label, prob) in enumerate(decoded, 1):
    print(f"{rank}. {label} ({prob:.4f})")

expected_label = decoded_orig[0][1]
new_labels = [label for (_, label, _) in decoded]
misclassified = expected_label not in new_labels
print(f"\n 是否判斷成功：{'✅ 是' if misclassified else '❌ 否'}（原為 {expected_label}）")

Image.fromarray(img_array.astype(np.uint8)).save("原始圖片.jpg")
Image.fromarray(adv_image_np).save("對抗樣本.jpg")
