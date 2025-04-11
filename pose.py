import cv2 as cv
import numpy as np

def select_img_from_video(video_file):
    # Open a video
    video = cv.VideoCapture(video_file)
    img_select = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        img_select.append(frame)
    video.release()
    return img_select
    

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # 이미지 수가 10개 이상이면 등간격으로 10개 선택
    if len(images) > 10:
        interval = len(images) // 10
        selected_images = [images[i] for i in range(0, len(images), interval)]
        selected_images = selected_images[:10]
    else:
        selected_images = images
    
    # Find 2D corner points from selected images
    img_points = []
    for img in selected_images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'There is no set of complete chessboard points!'
    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`
    # Calibrate the camera
    print('Start calibration...')
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)
    
    # Print calibration results
    print("\nCalibration Results:")
    print(f"RMS reprojection error: {ret}")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)

    return ret, mtx, dist, rvecs, tvecs

def draw_custom_image(images, K, dist_coeff, board_pattern, board_cellsize):
    # Load the cartoon image
    cartoon = cv.imread('cartoon.png', cv.IMREAD_UNCHANGED)  # Read with alpha channel
    if cartoon is None:
        raise FileNotFoundError("cartoon.png not found!")
    
    # Define the 4x4 region in chessboard coordinates (centered)
    center_x = board_pattern[0] // 2
    center_y = board_pattern[1] // 2
    
    # Define the 4x4 region points
    region_points = []
    for dy in range(-2, 2):
        for dx in range(-2, 2):
            region_points.append([center_x + dx, center_y + dy, 0])
    region_points = np.array(region_points, dtype=np.float32) * board_cellsize
    
    # numpy array and scale by cell size
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Initialize VideoWriter
    height, width = images[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 30.0, (width, height))
    
    # Process each image in the array
    for img in images:
        # Convert to grayscale for chessboard detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        complete, img_points = cv.findChessboardCorners(gray, board_pattern)
        
        if complete:
            # Estimate pose for this frame
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
            
            # Project 3D points to 2D using current frame's pose
            projected_points, _ = cv.projectPoints(region_points, rvec, tvec, K, dist_coeff)
            projected_points = np.int32(projected_points.reshape(-1, 2))
            
            # Calculate the perspective transform
            # Get the source points (original image corners)
            src_pts = np.array([[0, 0], [cartoon.shape[1], 0], 
                              [cartoon.shape[1], cartoon.shape[0]], [0, cartoon.shape[0]]], 
                              dtype=np.float32)
            # Get the 4 corners of the region
            top_left = projected_points[0]
            top_right = projected_points[3]
            bottom_left = projected_points[12]
            bottom_right = projected_points[15]
            
            # Get the destination points (projected points)
            dst_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            
            # Calculate the perspective transform matrix
            M = cv.getPerspectiveTransform(src_pts, dst_pts)
            
            # Warp the cartoon image
            warped = cv.warpPerspective(cartoon, M, (img.shape[1], img.shape[0]))
            
            # Overlay the cartoon
            overlay = img.copy()
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if warped[i, j, 2] > 0:  # Check alpha channel
                        overlay[i, j] = warped[i, j, :3]

            # Write frame to video
            out.write(overlay)
            
            # Show the result
            cv.imshow('Chessboard with Cartoon', overlay)
            
            # Press 'q' to quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    
    out.release()
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    board_pattern = (10, 7)
    board_cellsize = 25.0
    input_video = 'chess.avi'
    
    # 카메라 캘리브레이션 수행
    images = select_img_from_video(input_video)
    ret, mtx, dist, rvecs, tvecs = calib_camera_from_chessboard(images, board_pattern, board_cellsize)
    
    # Draw custom image
    draw_custom_image(images, mtx, dist, board_pattern, board_cellsize)
