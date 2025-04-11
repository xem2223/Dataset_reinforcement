import os
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 디렉토리 경로
img_dir = './eyecandies/docs/assets/images/objects'
img_files = sorted(os.listdir(img_dir))

# 앞에 5개 이미지 시각화
plt.figure(figsize=(12, 3))
for i in range(5):
    img_path = os.path.join(img_dir, img_files[i])
    img = Image.open(img_path)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(img_files[i])
plt.tight_layout()
plt.show(block=False)


# 경로 설정
gif_dir = './eyecandies/docs/assets/images/light_gifs'

# GIF 파일 5개 불러오기
gif_files = sorted([f for f in os.listdir(gif_dir) if f.endswith('.gif')])[:5]

# GIF마다 첫 5프레임씩 가져오기
all_gif_frames = []
for gif_file in gif_files:
    gif_path = os.path.join(gif_dir, gif_file)
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(len(frames))
            if len(frames) >= 5:
                break
    except EOFError:
        pass
    all_gif_frames.append(frames)

# 🎨 시각화 (5행 3열: 각 gif에서 프레임 5개)
plt.figure(figsize=(12, 10))

for row, frames in enumerate(all_gif_frames):
    for col, frame in enumerate(frames):
        idx = row * 5 + col + 1
        plt.subplot(5, 5, idx)
        plt.imshow(frame)
        plt.title(f'{gif_files[row]} - f{col+1}')
        plt.axis('off')

plt.tight_layout()
plt.show()