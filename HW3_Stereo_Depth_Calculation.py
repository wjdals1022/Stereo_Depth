import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches

import files_management

# Read the stereo-pair of images
img_left = files_management.read_left_image()
img_right = files_management.read_right_image()

# Use matplotlib to display the two images
_, image_cells = plt.subplots(1, 2, figsize=(20, 20))
image_cells[0].imshow(img_left)
image_cells[0].set_title('left image')
image_cells[1].imshow(img_right)
image_cells[1].set_title('right image')
plt.show()


# Read the calibration
p_left, p_right = files_management.get_projection_matrices()

# Use regular numpy notation instead of scientific one 
np.set_printoptions(suppress=True)

print("p_left \n", p_left)
print("\np_right \n", p_right)

def compute_left_disparity_map(img_left, img_right):
    
    
    # 찾을 수 있는 disparity의 최대 범위를 설정
    num_disparities = int(6*16*1) 
    #  매칭을 위한 블록 크기를 설정
    block_size = 10
    
    min_disparity = 0
    # 매칭을 위한 윈도우 크기
    window_size = 8
     
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Stereo SGBM matcher
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the left disparity map
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16
    
    return disp_left


# Compute the disparity map using the fuction above
disp_left = compute_left_disparity_map(img_left, img_right)

# Show the left disparity map
plt.figure(figsize=(10, 10))
plt.imshow(disp_left)
plt.show()


def decompose_projection_matrix(p):
    # 프로젝션 매트릭스가 3x4인지 확인
    assert p.shape == (3, 4), 
    
    # 3x3 부분 행렬 추출
    m = p[:, :3]

    k, r = np.linalg.qr(m)

    if np.linalg.det(k) < 0:
        k = -k
        r = -r

    t = np.linalg.inv(k) @ p[:, 3]
    

    t = np.append(t, 1)  # [x, y, z] + 1 (동차 좌표)
    
    return k, r, t

# Decompose each matrix
k_left, r_left, t_left = decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_projection_matrix(p_right)

# Display the matrices
print("k_left \n", k_left)
print("\nr_left \n", r_left)
print("\nt_left \n", t_left)
print("\nk_right \n", k_right)
print("\nr_right \n", r_right)
print("\nt_right \n", t_right)


def calc_depth_map(disp_left, k_left, t_left, t_right):
    ### START CODE HERE ###
    
    f = k_left[0, 0]  # 초점 거리 f는 K 행렬의 (0,0) 위치
    
    # 두 카메라 간의 거리는 t_left와 t_right의 차이로 계산
    b = np.linalg.norm(t_left - t_right)  # 두 카메라의 변환 벡터 간의 차이로 기준선 계산
    

    disp_left[disp_left == 0] = 0.1  # 0인 disparity를 작은 값으로 변경
    disp_left[disp_left == -1] = 0.1  # -1인 disparity를 작은 값으로 변경


    depth_map = np.zeros_like(disp_left)  # disparity 맵과 같은 크기로 depth 맵 초기화


    depth_map = (f * b) / disp_left  # 깊이 계산
    
    ### END CODE HERE ###
    
    return depth_map


# Get the depth map by calling the above function
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

# Display the depth map
plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(depth_map_left, cmap='flag')
plt.show()