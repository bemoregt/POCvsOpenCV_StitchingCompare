import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
## computer vision package using frequency domain
import openfv as fv

def fourier_stitch_robust(img1, img2):
    """Phase Only Correlation을 이용한 강건한 이미지 스티칭"""
    # POC를 사용하여 상대적 위치 계산
    dy, dx, _ = fv.ww_phase_only_correlation(img1, img2)
    
    # 정수 좌표로 변환 (반올림)
    dy, dx = int(round(dy)), int(round(dx))
    
    # 이미지 크기
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 이동량에 따라 새 이미지 크기 계산
    if dx >= 0 and dy >= 0:
        H = max(h1, dy + h2)
        W = max(w1, dx + w2)
        offset_y1, offset_x1 = 0, 0
        offset_y2, offset_x2 = dy, dx
    elif dx >= 0 and dy < 0:
        H = max(h1 - dy, h2)
        W = max(w1, dx + w2)
        offset_y1, offset_x1 = -dy, 0
        offset_y2, offset_x2 = 0, dx
    elif dx < 0 and dy >= 0:
        H = max(h1, dy + h2)
        W = max(w1 - dx, w2)
        offset_y1, offset_x1 = 0, -dx
        offset_y2, offset_x2 = dy, 0
    else:  # dx < 0 and dy < 0
        H = max(h1 - dy, h2)
        W = max(w1 - dx, w2)
        offset_y1, offset_x1 = -dy, -dx
        offset_y2, offset_x2 = 0, 0
    
    # 새 이미지 생성
    stitched_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 첫 번째 이미지 배치
    stitched_img[offset_y1:offset_y1+h1, offset_x1:offset_x1+w1] = img1
    
    # 두 번째 이미지와 겹치는 부분을 평균값으로 설정 (블렌딩)
    for y in range(h2):
        for x in range(w2):
            y_stitch = y + offset_y2
            x_stitch = x + offset_x2
            
            if 0 <= y_stitch < H and 0 <= x_stitch < W:
                # 겹치는 영역 확인
                if np.any(stitched_img[y_stitch, x_stitch] > 0) and np.any(img2[y, x] > 0):
                    # 알파 블렌딩 (50:50)
                    stitched_img[y_stitch, x_stitch] = (0.5 * stitched_img[y_stitch, x_stitch] + 0.5 * img2[y, x]).astype(np.uint8)
                elif np.any(img2[y, x] > 0):
                    stitched_img[y_stitch, x_stitch] = img2[y, x]
    
    return stitched_img

def opencv_stitch(img1, img2):
    """OpenCV의 특징점 기반 이미지 스티칭"""
    try:
        # SIFT 특징점 검출기 생성
        sift = cv2.SIFT_create()
        
        # 그레이스케일 이미지 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 특징점 및 디스크립터 추출
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # 특징점 매칭
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # 좋은 매칭 선택 (Lowe's ratio test)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # 최소 4개의 좋은 매칭이 필요함
        if len(good_matches) < 4:
            return None
            
        # 매칭된 특징점 좌표 추출
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 호모그래피 계산
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 이미지 변환
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts1, H)
        
        # 전체 이미지 영역 계산
        pts = np.concatenate((pts2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        
        # 변환 행렬 조정
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        
        # 이미지 워핑
        result = cv2.warpPerspective(img1, Ht @ H, (xmax-xmin, ymax-ymin))
        
        # 두 번째 이미지 추가
        result[-ymin:h2-ymin, -xmin:w2-xmin] = img2
        
        return result
    except Exception as e:
        print(f"OpenCV 스티칭 오류: {e}")
        return None

# 이미지에 블러 효과 적용
def apply_blur(image, kernel_size):
    if kernel_size <= 1:
        return image.copy()
    # 커널 사이즈를 홀수로 만들기
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 이미지에 노이즈 효과 적용 (백색 가우시안 노이즈)
def apply_noise(image, noise_level):
    if noise_level <= 0:
        return image.copy()
    
    # 이미지를 float32로 변환
    img_float = image.astype(np.float32)
    
    # 백색 가우시안 노이즈 생성 (모든 채널에 동일한 노이즈 적용)
    h, w, c = image.shape
    noise = np.random.normal(0, noise_level, (h, w)).astype(np.float32)
    
    # 노이즈를 각 채널에 적용
    for i in range(c):
        img_float[:,:,i] = img_float[:,:,i] + noise
    
    # 값 범위를 0-255로 클리핑
    img_float = np.clip(img_float, 0, 255)
    
    # uint8로 변환하여 반환
    return img_float.astype(np.uint8)

class ImageStitchingApp:
    def __init__(self, root, img1_path, img2_path):
        self.root = root
        self.root.title("Image Stitching Application")
        
        # 이미지 경로 설정
        self.img1_path = img1_path
        self.img2_path = img2_path
        
        # 이미지 불러오기
        self.img1_original = cv2.imread(self.img1_path)
        self.img2_original = cv2.imread(self.img2_path)
        
        if self.img1_original is None or self.img2_original is None:
            print("It can't be load image. Watch your path.")
            root.destroy()
            return
            
        self.img1_processed = self.img1_original.copy()
        self.img2_processed = self.img2_original.copy()
        self.stitched_img_poc = None
        self.stitched_img_opencv = None
        
        # 다크 테마 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use('clam')
        bg_color = '#2E2E2E'
        fg_color = '#FFFFFF'
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabelframe', background=bg_color, foreground=fg_color)
        self.style.configure('TLabelframe.Label', background=bg_color, foreground=fg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TButton', background='#4E4E4E', foreground=fg_color)
        self.style.map('TButton', background=[('active', '#5E5E5E')])
        
        # 전체 창 배경색 설정
        self.root.configure(background=bg_color)
        
        # 메인 프레임
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 이미지 프레임
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=0, column=0, columnspan=3, pady=10)
        
        # 이미지 1 프레임
        img1_frame = ttk.LabelFrame(self.image_frame, text="Image 1")
        img1_frame.grid(row=0, column=0, padx=10)
        
        self.img1_label = ttk.Label(img1_frame)
        self.img1_label.grid(row=0, column=0)
        
        # 블러 콤보박스
        ttk.Label(img1_frame, text="Blur Kernel Size").grid(row=1, column=0, pady=5)
        self.blur_kernel = tk.StringVar()
        blur_combo = ttk.Combobox(img1_frame, textvariable=self.blur_kernel, 
                                   values=[str(i) for i in range(3, 50, 2)])
        blur_combo.grid(row=2, column=0, pady=5)
        blur_combo.current(0)  # 기본값 설정
        blur_combo.bind("<<ComboboxSelected>>", self.update_processed_image1)
        
        # 이미지 2 프레임
        img2_frame = ttk.LabelFrame(self.image_frame, text="Image 2")
        img2_frame.grid(row=0, column=1, padx=10)
        
        self.img2_label = ttk.Label(img2_frame)
        self.img2_label.grid(row=0, column=0)
        
        # 노이즈 콤보박스
        ttk.Label(img2_frame, text="Noise Level (%)").grid(row=1, column=0, pady=5)
        self.noise_level = tk.StringVar()
        noise_combo = ttk.Combobox(img2_frame, textvariable=self.noise_level, 
                                   values=[str(i) for i in range(0, 291, 20)])
        noise_combo.grid(row=2, column=0, pady=5)
        noise_combo.current(0)  # 기본값 설정
        noise_combo.bind("<<ComboboxSelected>>", self.update_processed_image2)

        # 결과 프레임 (2x1 그리드로 변경)
        result_frame = ttk.LabelFrame(self.image_frame, text="Stitch Result")
        result_frame.grid(row=0, column=2, padx=10)
        
        # POC 결과
        poc_frame = ttk.LabelFrame(result_frame, text="POC Method")
        poc_frame.grid(row=0, column=0, padx=5, pady=5)
        
        self.poc_label = ttk.Label(poc_frame)
        self.poc_label.grid(row=0, column=0)
        
        # OpenCV 결과
        opencv_frame = ttk.LabelFrame(result_frame, text="OpenCV Method")
        opencv_frame.grid(row=1, column=0, padx=5, pady=5)
        
        self.opencv_label = ttk.Label(opencv_frame)
        self.opencv_label.grid(row=0, column=0)
        
        # 스티칭 버튼
        ttk.Button(main_frame, text="Stitch", command=self.stitch_images).grid(row=1, column=1, pady=10)
        
        # 이미지 크기 조정 비율
        self.resize_ratio = 0.5
        
        # 초기 이미지 표시
        self.display_image(self.img1_original, self.img1_label)
        self.display_image(self.img2_original, self.img2_label)
    
    def update_processed_image1(self, event):
        try:
            kernel_size = int(self.blur_kernel.get())
            self.img1_processed = apply_blur(self.img1_original, kernel_size)
            self.display_image(self.img1_processed, self.img1_label)
        except Exception as e:
            print(f"Blurring Error: {e}")
    
    def update_processed_image2(self, event):
        try:
            noise_level = int(self.noise_level.get())
            self.img2_processed = apply_noise(self.img2_original, noise_level * 2.55)  # 0-90% 범위를 0-229.5로 변환
            self.display_image(self.img2_processed, self.img2_label)
        except Exception as e:
            print(f"Noising Error: {e}")
    
    def stitch_images(self):
        try:
            # POC 방식 스티칭
            self.stitched_img_poc = fourier_stitch_robust(self.img1_processed, self.img2_processed)
            self.display_image(self.stitched_img_poc, self.poc_label)
            
            # OpenCV 방식 스티칭
            self.stitched_img_opencv = opencv_stitch(self.img1_processed, self.img2_processed)
            if self.stitched_img_opencv is not None:
                self.display_image(self.stitched_img_opencv, self.opencv_label)
            else:
                # 스티칭 실패 시 기본 이미지 표시
                placeholder = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(placeholder, "OpenCV Stitching Failed", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.display_image(placeholder, self.opencv_label)
        except Exception as e:
            print(f"Stitching Error: {e}")
    
    def display_image(self, img, label):
        if img is None:
            return
        
        # BGR에서 RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 조정
        height, width = img_rgb.shape[:2]
        new_height = int(height * self.resize_ratio)
        new_width = int(width * self.resize_ratio)
        img_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # PIL 이미지로 변환
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # 라벨에 이미지 표시
        label.configure(image=img_tk)
        label.image = img_tk

# 메인 애플리케이션 실행
if __name__ == "__main__":
    # 이미지 경로 설정 (여기서 원하는 경로로 변경)
    img1_path = "image1.png"
    img2_path = "image2.png"
    
    root = tk.Tk()
    app = ImageStitchingApp(root, img1_path, img2_path)
    root.mainloop()