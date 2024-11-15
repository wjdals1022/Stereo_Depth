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
    #  매칭을 위한 블록 크기를 설정합니다. 작은 값일수록 더 세밀한 매칭을 할 수 있지만, 노이즈에 민감하고 계산이 더 복잡해집니다. 
    #  큰 값일수록 안정적이지만, 세부적인 특징을 놓칠 수 있습니다.
    block_size = 15
    
    min_disparity = 0
    # 매칭을 위한 윈도우 크기입니다. block_size와 유사하게, 이 값이 커지면 계산은 빨라지지만, 세부적인 차이를 잘 잡아내지 못할 수 있습니다.
    # 작은 window_size는 더 세밀한 결과를 제공하지만, 계산 시간이 오래 걸리고, 노이즈에 민감할 수 있습니다.
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
    assert p.shape == (3, 4), "프로젝션 매트릭스는 3x4 크기여야 합니다."
    
    # 3x3 부분 행렬 추출
    m = p[:, :3]
    
    # QR 분해를 사용하여 내재 행렬 K와 회전 행렬 R 얻기
    k, r = np.linalg.qr(m)
    
    # 내재 행렬 K의 대각선 원소가 음수일 경우 K의 부호를 반전
    if np.linalg.det(k) < 0:
        k = -k
        r = -r
    
    # 변환 벡터 t는 K의 역행렬과 마지막 열을 곱하여 계산
    t = np.linalg.inv(k) @ p[:, 3]
    
    # 변환 벡터의 형식을 맞추기 위해 4차원 벡터로 변환
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
    
    # Get the focal length from the K matrix (K는 내재 행렬이므로, (0,0) 또는 (1,1) 위치에서 초점 거리를 찾을 수 있음)
    f = k_left[0, 0]  # 초점 거리 f는 K 행렬의 (0,0) 위치에 있음
    
    # Get the distance between the cameras from the t matrices (baseline)
    # 두 카메라 간의 거리는 t_left와 t_right의 차이로 계산됨
    b = np.linalg.norm(t_left - t_right)  # 두 카메라의 변환 벡터 간의 차이로 기준선 계산
    
    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1  # 0인 disparity를 작은 값으로 변경
    disp_left[disp_left == -1] = 0.1  # -1인 disparity를 작은 값으로 변경

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.zeros_like(disp_left)  # disparity 맵과 같은 크기로 depth 맵 초기화

    # Calculate the depths using the formula: depth = (f * b) / disparity
    depth_map = (f * b) / disp_left  # 깊이 계산
    
    ### END CODE HERE ###
    
    return depth_map


# Get the depth map by calling the above function
depth_map_left = calc_depth_map(disp_left, k_left, t_left, t_right)

# Display the depth map
plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(depth_map_left, cmap='flag')
plt.show()