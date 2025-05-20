import cv2
import cv2.aruco as aruco
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

def create_aruco_grid_pdf(output_path="aruco_grid.pdf",
                         markers_x=5, markers_y=9,
                         marker_size_mm=50,
                         margin_size_mm=10):
    """
    Create PDF with a grid of ArUco markers (no checkerboard pattern)
    """
    # Calculate total board dimensions
    total_width = markers_x * (marker_size_mm + margin_size_mm) - margin_size_mm
    total_height = markers_y * (marker_size_mm + margin_size_mm) - margin_size_mm
    
    # Create ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # Create white background image
    img = np.ones((int(total_height), int(total_width)), dtype=np.uint8) * 255
    
    # Generate and place markers
    marker_id = 0
    for y in range(markers_y):
        for x in range(markers_x):
            # Create marker - this is the modern way
            marker_img = aruco.drawMarker(aruco_dict, marker_id, int(marker_size_mm))
            
            # Calculate position
            x_pos = int(x * (marker_size_mm + margin_size_mm))
            y_pos = int(y * (marker_size_mm + margin_size_mm))
            
            # Place marker on image
            img[y_pos:y_pos+marker_size_mm, x_pos:x_pos+marker_size_mm] = marker_img
            
            marker_id += 1
            if marker_id >= 50:  # For DICT_4X4_50
                marker_id = 0
    
    # Save temporary image
    cv2.imwrite("temp_aruco.png", img)
    
    # Create PDF centered on A4
    c = canvas.Canvas(output_path, pagesize=A4)
    a4_width, a4_height = A4
    
    # Calculate position to center the board
    x_pos = (a4_width - total_width) / 2
    y_pos = (a4_height - total_height) / 2
    
    # Add board image
    c.drawImage("temp_aruco.png", x_pos, y_pos, 
               width=total_width, height=total_height)
    
    # Add border and info text
    c.rect(x_pos, y_pos, total_width, total_height)
    c.setFont("Helvetica", 10)
    c.drawString(30, 30, f"ArUco Grid - {markers_x}x{markers_y} markers, {marker_size_mm}mm each")
    
    c.save()
    print(f"PDF saved to {output_path} with centered {total_width}x{total_height}mm grid")

if __name__ == "__main__":
    create_aruco_grid_pdf(markers_x=5, markers_y=7, marker_size_mm=50, margin_size_mm=50)