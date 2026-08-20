import cv2
import numpy as np

def auto_annotate_lesions(image_path, output_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像，请检查路径是否正确: {image_path}")
        return

    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建有效眼底区域掩码（剔除四周的纯黑背景）
    _, roi_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # ==========================================
    # 目标 A：硬性渗出 (Hard Exudates) - 极度严格
    # ==========================================
    # 提高亮度阈值到 230，只抓最亮的白斑
    _, bright_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    contours_bright, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_bright:
        area = cv2.contourArea(cnt)
        # 缩小面积上限，避免框到大面积的背景高光
        if 15 < area < 150:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # ==========================================
    # 目标 B：微动脉瘤与出血点 - 极度严格防血管
    # ==========================================
    # 降低暗度阈值到 40，只抓死黑的像素
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.bitwise_and(dark_mask, roi_mask)

    contours_dark, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_dark:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = float(w) / h if h != 0 else 0
        # 1. 面积进一步缩小 (血管面积通常很大)
        # 2. 长宽比极其严苛 (0.85 到 1.15)，逼近正方形/圆形，彻底排查长条形血管
        if 3 < area < 20 and 0.85 < aspect_ratio < 1.15:
            # 避开中央巨大的黄斑区（Macula）
            img_h, img_w = img.shape[:2]
            center_x, center_y = img_w // 2, img_h // 2
            if not (center_x - 100 < x < center_x + 50 and center_y - 80 < y < center_y + 80):
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # 3. 保存结果
    cv2.imwrite(output_path, img_copy)
    print(f"✅ 自动标注完成，已保存至: {output_path}")


# ==========================================
# 运行执行 (完全使用你的绝对路径)
# ==========================================
if __name__ == '__main__':
    input_img = r'E:\MsC Project\Ben Graham_img\bengraham_image.jpg'
    output_img = r'E:\MsC Project\Ben Graham_img\Ben Graham_lighting.jpg'

    auto_annotate_lesions(input_img, output_img)