import cv2
import numpy as np
import chess
import chess.engine
from ultralytics import YOLO
from PIL import Image
from google import genai
import tkinter as tk

# ==========================================
# 1. SETTINGS AND PATHS
# ==========================================
# Update this path to where Stockfish is located on your local machine
STOCKFISH_PATH = r"C:\path\to\your\stockfish\stockfish-windows-x86-64-avx2.exe" 

# Use 0 for default webcam, or replace with your IP webcam address 
CAMERA_ADDRESS = "http://YOUR_PHONE_IP:8080/video"

YOLO_MODEL_PATH = "chess-model-yolov8m.pt" 
try:
    print("Loading Local AI (YOLO)...")
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"ERROR: YOLO model not found! {e}")

def create_yolo_dict(model_names):
    dictionary = {}
    for obj_id, name in model_names.items():
        name_str = name.lower()
        letter = ''
        if 'bishop' in name_str: letter = 'b'
        elif 'king' in name_str: letter = 'k'
        elif 'knight' in name_str: letter = 'n'
        elif 'pawn' in name_str: letter = 'p'
        elif 'queen' in name_str: letter = 'q'
        elif 'rook' in name_str: letter = 'r'
        if 'white' in name_str: letter = letter.upper()
        dictionary[obj_id] = letter
    return dictionary
YOLO_CLASS_LETTERS = create_yolo_dict(model.names)

# IMPORTANT: Never share your actual API key publicly!
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" 
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("Cloud AI (Gemini) Connection Ready!")
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini Client! {e}")

# ==========================================
# 2. TRACKING AND MARGIN SETTINGS
# ==========================================
system_mode = 'AUTO' 
corner_points = []
current_matrix_M = None

lk_params = dict(winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray_frame = None 
p0 = None            

MARGIN = 60 # 60 pixels breathing room for pieces (prevents heads from being cropped)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect

def mouse_click(event, x, y, flags, param):
    global corner_points, system_mode, p0, old_gray_frame, gray_frame
    if event == cv2.EVENT_LBUTTONDOWN and system_mode == 'MANUAL':
        if len(corner_points) < 4:
            corner_points.append((x, y))
            print(f"Point added: ({x}, {y}) - Total: {len(corner_points)}/4")
            if len(corner_points) == 4:
                pts = np.array(corner_points, dtype="float32")
                pts = order_points(pts)
                p0 = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                old_gray_frame = gray_frame.copy()
                system_mode = 'TRACKING'
                print("--- 4 POINTS SET, TRACKING STARTED ---")

# ==========================================
# 3. LOCAL, CLOUD, AND GUI FUNCTIONS
# ==========================================

# Unicode Chess Symbols for GUI
PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
    '': ' '
}

def fen_to_board(fen):
    board = [['' for _ in range(8)] for _ in range(8)]
    rows = fen.split(' ')[0].split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                board[r][c] = char
                c += 1
    return board

def board_to_fen(board):
    fen_rows = []
    for row in board:
        empty = 0
        fen_row = ""
        for square in row:
            if square == '':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += square
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w KQkq - 0 1"

def visual_fen_corrector(initial_fen):
    """
    Displays an interactive chessboard to the user,
    allows manual correction of errors, and returns the approved FEN.
    """
    approved_fen = initial_fen
    
    root = tk.Tk()
    root.title("YOLO Error Correction Panel")
    root.attributes('-topmost', True)
    
    try:
        board_matrix = fen_to_board(initial_fen)
    except Exception:
        board_matrix = [['' for _ in range(8)] for _ in range(8)]
        
    buttons = [[None for _ in range(8)] for _ in range(8)]
    selected_button = [None] 
    
    # --- Top Palette (Piece Selection) ---
    palette_frame = tk.Frame(root, bg="#2b2b2b")
    palette_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    
    tk.Label(palette_frame, text="1. Select the square you want to correct on the board.\n2. Choose a new piece below or click 'X' to clear it.", 
             bg="#2b2b2b", fg="white", font=("Arial", 12)).pack(pady=5)
    
    piece_btn_frame = tk.Frame(palette_frame, bg="#2b2b2b")
    piece_btn_frame.pack()
    
    def select_piece(letter):
        if selected_button[0] is not None:
            r, c = selected_button[0]
            board_matrix[r][c] = letter
            buttons[r][c].config(text=PIECE_UNICODE[letter])
            
            for i in range(8):
                for j in range(8):
                    color = "#F0D9B5" if (i+j)%2==0 else "#B58863"
                    buttons[i][j].config(bg=color)
            selected_button[0] = None

    for letter, uni in PIECE_UNICODE.items():
        text = uni if uni != ' ' else 'X (CLEAR)'
        fg_color = "black" if letter.isupper() else "blue" 
        if text == 'X (CLEAR)': fg_color = "red"
        
        b = tk.Button(piece_btn_frame, text=text, font=("Arial", 18), width=4, fg=fg_color,
                      command=lambda h=letter: select_piece(h))
        b.pack(side=tk.LEFT, padx=2)

    # --- Chess Board UI ---
    board_frame = tk.Frame(root)
    board_frame.pack(pady=10)
    
    def click_square(r, c):
        selected_button[0] = (r, c)
        for i in range(8):
            for j in range(8):
                color = "#F0D9B5" if (i+j)%2==0 else "#B58863"
                buttons[i][j].config(bg=color)
        buttons[r][c].config(bg="#FFCE33") # Highlight selected square
        
    for r in range(8):
        for c in range(8):
            color = "#F0D9B5" if (r+c)%2==0 else "#B58863"
            letter = board_matrix[r][c]
            fg_color = "black" if letter.isupper() else "blue"
            
            btn = tk.Button(board_frame, text=PIECE_UNICODE[letter], font=("Segoe UI Symbol", 24), 
                            width=3, height=1, bg=color, fg=fg_color,
                            command=lambda r=r, c=c: click_square(r, c))
            btn.grid(row=r, column=c)
            buttons[r][c] = btn

    # --- Approve Button ---
    def approve():
        nonlocal approved_fen
        approved_fen = board_to_fen(board_matrix)
        root.destroy()
        
    tk.Button(root, text="FINISH CORRECTION AND APPROVE", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", 
              width=30, height=2, command=approve).pack(pady=10)
    
    root.mainloop()
    return approved_fen

def generate_fen_from_yolo(original_img, matrix_M):
    board_matrix = [['' for _ in range(8)] for _ in range(8)]
    confidence_matrix = [[0.0 for _ in range(8)] for _ in range(8)] 
    
    results = model.predict(original_img, conf=0.45, iou=0.45, verbose=False)
    
    for box in results[0].boxes:
        x_center = float(box.xywh[0][0])
        y_center = float(box.xywh[0][1])
        h = float(box.xywh[0][3])
        class_id = int(box.cls[0])
        confidence_score = float(box.conf[0])
        
        base_x = x_center
        base_y = y_center + (h * 0.35) 
        piece_letter = YOLO_CLASS_LETTERS.get(class_id, '?')

        point = np.array([[[base_x, base_y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, matrix_M)
        
        bx, by = transformed_point[0][0][0], transformed_point[0][0][1]
        
        if MARGIN <= bx < MARGIN + 400 and MARGIN <= by < MARGIN + 400:
            col = int((bx - MARGIN) // 50)
            row = int((by - MARGIN) // 50)
            if confidence_score > confidence_matrix[row][col]:
                board_matrix[row][col] = piece_letter
                confidence_matrix[row][col] = confidence_score

    return board_to_fen(board_matrix)

def get_fen_from_gemini(original_img, matrix_M):
    total_size = 400 + (MARGIN * 2)
    clean_warped = cv2.warpPerspective(original_img, matrix_M, (total_size, total_size))
    
    rgb_board = cv2.cvtColor(clean_warped, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_board)
    
    prompt = """
    You are an expert chess engine vision module. 
    There is a perspective-corrected chessboard in the image below. 
    ONLY return the standard FEN code of the board setup. 
    You are viewing the board from white's perspective (the bottom row is the 1st rank).
    Never provide explanations, do not use Markdown. Just give the FEN code.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_image]
        )
        fen_code = response.text.strip().replace("```fen", "").replace("```", "").strip()
        return fen_code
    except Exception as e:
        print(f"API Connection Error: {e}")
        return None

def is_position_valid(fen_code):
    try:
        board = chess.Board(fen_code)
        return board.is_valid()
    except:
        return False

def find_best_move(fen_code):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        board = chess.Board(fen_code)
        result = engine.play(board, chess.engine.Limit(time=0.5)) 
        best_move = result.move
        engine.quit()
        return str(best_move).upper()
    except Exception as e:
        return f"Stockfish Error: {e}"

# ==========================================
# 4. MAIN LOOP AND CAMERA
# ==========================================
cap = cv2.VideoCapture(CAMERA_ADDRESS)

cv2.namedWindow('Original Camera', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Camera', 640, 480)
cv2.setMouseCallback('Original Camera', mouse_click)

while True:
    ret, frame = cap.read()
    if not ret: break

    clean_frame = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if system_mode == 'AUTO':
        cv2.putText(frame, "[AUTO] L: Lock | R: Manual Select", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] 

        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 30000:  
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    if 0.5 <= float(w) / h <= 1.5:
                        best_cnt = approx
                        break 
        if best_cnt is not None:
            cv2.drawContours(frame, [best_cnt], -1, (0, 255, 0), 4)

    elif system_mode == 'MANUAL':
        cv2.putText(frame, "[MANUAL] Click 4 Corners", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        for point in corner_points:
            cv2.circle(frame, point, 8, (0, 255, 255), -1)

    elif system_mode == 'TRACKING':
        cv2.putText(frame, "[TRACKING LOCKED] C: Hybrid Analysis | R: Reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray_frame, gray_frame, p0, None, **lk_params)
        
        if p1 is not None and len(p1[st==1]) == 4:
            p0 = p1.copy()
            old_gray_frame = gray_frame.copy()
            pts_current = p0.reshape(4, 2)
            
            dst = np.array([
                [MARGIN, MARGIN], 
                [MARGIN + 399, MARGIN], 
                [MARGIN + 399, MARGIN + 399], 
                [MARGIN, MARGIN + 399]
            ], dtype="float32")
            
            current_matrix_M = cv2.getPerspectiveTransform(pts_current, dst)
            pts_draw = np.int32(pts_current).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_draw], isClosed=True, color=(0, 255, 0), thickness=3)
        else:
            cv2.putText(frame, "TRACKING LOST! Press R to reset.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Original Camera', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l') and system_mode == 'AUTO':
        if best_cnt is not None:
            pts = best_cnt.reshape(4, 2)
            pts = order_points(pts)
            p0 = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            old_gray_frame = gray_frame.copy()
            system_mode = 'TRACKING'
    elif key == ord('r'):
        corner_points = []
        current_matrix_M = None
        p0 = None
        system_mode = 'MANUAL'
    elif key == ord('c'):
        if system_mode == 'TRACKING' and current_matrix_M is not None:
            print("\n--- HYBRID ANALYSIS STARTING ---")
            
            print("[STAGE 1] YOLO (Local) is analyzing...")
            yolo_fen = generate_fen_from_yolo(clean_frame, current_matrix_M)
            
            print(">> Visual Correction Panel opened. Please correct errors on the board and approve.")
            approved_fen = visual_fen_corrector(yolo_fen)
            
            print(f"APPROVED FEN: {approved_fen}")
            
            if is_position_valid(approved_fen):
                print(">> Position valid! Sending to Stockfish...")
                best_move = find_best_move(approved_fen)
                print(f">>> RECOMMENDED MOVE: {best_move} <<<")
                print("------------------------\n")
            else:
                print(">> ERROR: Manually corrected board still violates chess rules! (e.g., missing king)")
                print("------------------------\n")

cap.release()
cv2.destroyAllWindows()