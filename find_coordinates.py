import cv2

def find_line_coordinates(video_path="test_video2.mp4"):
    """Click on video to get pixel coordinates for counting lines"""
    
    # Store clicked points
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the actual coordinates on the original frame
            # Account for the scaling
            actual_x = int(x * scale_factor)
            actual_y = int(y * scale_factor)
            
            points.append((actual_x, actual_y))
            print(f"Point {len(points)}: ({actual_x}, {actual_y})")
            
            # Draw on display frame (scaled)
            cv2.circle(display_frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(display_frame, f"P{len(points)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # If we have 2 points, draw the line
            if len(points) == 2:
                # Draw line on display frame (scaled positions)
                pt1 = (int(points[0][0] / scale_factor), int(points[0][1] / scale_factor))
                pt2 = (int(points[1][0] / scale_factor), int(points[1][1] / scale_factor))
                cv2.line(display_frame, pt1, pt2, (0, 0, 255), 2)
                
                print(f"\n{'='*60}")
                print(f"Line coordinates: [{points[0]}, {points[1]}]")
                print(f"{'='*60}")
                print(f"Add this to your COUNT_LINES:")
                print(f"    [{points[0]}, {points[1]}],")
                print(f"{'='*60}\n")
                points.clear()  # Reset for next line
            
            cv2.imshow("Frame - Click to mark counting lines", display_frame)
    
    # Open video and get first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot read video!")
        return
    
    # Get original frame dimensions
    orig_height, orig_width = frame.shape[:2]
    print(f"Original video resolution: {orig_width}x{orig_height}")
    
    # Calculate scale to fit screen (assuming 1920x1080 max, adjust as needed)
    max_width = 1280  # Adjust this to your screen width
    max_height = 720  # Adjust this to your screen height
    
    # Calculate scaling factor to fit within max dimensions
    width_scale = max_width / orig_width
    height_scale = max_height / orig_height
    scale_factor_inv = min(width_scale, height_scale)
    scale_factor = 1.0 / scale_factor_inv
    
    # Calculate new dimensions
    display_width = int(orig_width * scale_factor_inv)
    display_height = int(orig_height * scale_factor_inv)
    
    print(f"Display window size: {display_width}x{display_height}")
    print(f"Scale factor: {scale_factor:.2f}x\n")
    
    # Create resizable window
    cv2.namedWindow("Frame - Click to mark counting lines", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame - Click to mark counting lines", display_width, display_height)
    cv2.setMouseCallback("Frame - Click to mark counting lines", mouse_callback)
    
    print("="*60)
    print("Instructions:")
    print("="*60)
    print("1. Click TWO points to define a line:")
    print("   - First click: LEFT endpoint of the line")
    print("   - Second click: RIGHT endpoint of the line")
    print("2. The line will be drawn in RED")
    print("3. Press 'n' for next frame")
    print("4. Press 'r' to reset/clear current points")
    print("5. Press 'q' to quit")
    print("6. Copy the printed coordinates to your COUNT_LINES")
    print("="*60 + "\n")
    
    # Create display frame (scaled)
    display_frame = cv2.resize(frame, (display_width, display_height))
    cv2.imshow("Frame - Click to mark counting lines", display_frame)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('n'):
            # Next frame
            ret, frame = cap.read()
            if ret:
                display_frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow("Frame - Click to mark counting lines", display_frame)
            else:
                print("End of video")
                break
        elif key == ord('r'):
            # Reset points
            points.clear()
            display_frame = cv2.resize(frame, (display_width, display_height))
            cv2.imshow("Frame - Click to mark counting lines", display_frame)
            print("Points reset\n")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LINE COORDINATE FINDER")
    print("="*60 + "\n")
    find_line_coordinates("test_video2.mp4")
