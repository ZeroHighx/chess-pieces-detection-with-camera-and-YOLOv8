import cv2
import numpy as np
import pygame
import sys
import os
import urllib.request
from ultralytics import YOLO
from stockfish import Stockfish

# --- Configuration ---
CAMERA_URL = "http://YOUR_PHONE_IP:8080/video"  # Use 0 for local webcam
MODEL_PATH = "chess-model-yolov8m.pt"
STOCKFISH_PATH = r"C:\path\to\your\stockfish\stockfish-windows-x86-64-avx2.exe"

# Global states
board_points = []

# --- Asset Maps ---
PIECE_URLS = {
    'r': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/br.png',
    'n': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bn.png',
    'b': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bb.png',
    'q': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bq.png',
    'k': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bk.png',
    'p': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/bp.png',
    'R': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wr.png',
    'N': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wn.png',
    'B': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wb.png',
    'Q': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wq.png',
    'K': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wk.png',
    'P': 'https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png'
}

PIECE_FILES = {k: f"{'b' if k.islower() else 'w'}_{k.lower()}.png" for k in PIECE_URLS.keys()}


def init_models():
    """Loads YOLO and Stockfish models."""
    print("[INFO] Loading models, please wait...")
    try:
        model = YOLO(MODEL_PATH)
        engine = Stockfish(path=STOCKFISH_PATH)
        print("[INFO] Models loaded successfully.")
        return model, engine
    except Exception as e:
        print(f"[FATAL] Failed to initialize models: {e}")
        sys.exit(1)


def fetch_assets():
    """Downloads missing chess piece images."""
    asset_dir = "pieces"
    os.makedirs(asset_dir, exist_ok=True)

    missing = [f for f in PIECE_FILES.values() if not os.path.exists(os.path.join(asset_dir, f))]
    
    if missing:
        print("[INFO] Downloading missing chess piece assets...")
        for fen_char, url in PIECE_URLS.items():
            filepath = os.path.join(asset_dir, PIECE_FILES[fen_char])
            if not os.path.exists(filepath):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"[WARNING] Failed to fetch {fen_char}: {e}")


def mouse_click_handler(event, x, y, flags, params):
    """Stores user clicks for manual board cropping."""
    global board_points
    if event == cv2.EVENT_LBUTTONDOWN and len(board_points) < 4:
        board_points.append([x, y])


def detect_board_contours(frame):
    """Finds the largest 4-point contour to auto-crop the board."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10000:
        return None

    approx = cv2.approxPolyDP(largest, 0.05 * cv2.arcLength(largest, True), True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    return None


def apply_perspective_transform(frame, pts):
    """Warps the selected points into a top-down square image."""
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    w_a = np.linalg.norm(br - bl)
    w_b = np.linalg.norm(tr - tl)
    max_w = max(int(w_a), int(w_b))

    h_a = np.linalg.norm(tr - br)
    h_b = np.linalg.norm(tl - bl)
    max_h = max(int(h_a), int(h_b))

    dst = np.array([
        [0, 0], [max_w - 1, 0], 
        [max_w - 1, max_h - 1], [0, max_h - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, matrix, (max_w, max_h))


def parse_yolo_predictions(results, img_shape):
    """Converts YOLO bounding boxes into an 8x8 chess matrix."""
    board = [["" for _ in range(8)] for _ in range(8)]
    h, w = img_shape[:2]
    sq_h, sq_w = h / 8, w / 8

    # Clean map to handle different label formatting
    color_map = {'black': 'b', 'white': 'w'}
    piece_map = {'rook': 'r', 'knight': 'n', 'bishop': 'b', 'queen': 'q', 'king': 'k', 'pawn': 'p'}

    for box in results[0].boxes:
        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
        class_name = results[0].names[int(box.cls[0])].lower().replace('-', ' ').replace('_', ' ')
        parts = class_name.split()

        char = ""
        if len(parts) == 2 and parts[0] in color_map and parts[1] in piece_map:
            char = piece_map[parts[1]]
            char = char if parts[0] == 'black' else char.upper()
        elif len(parts) == 1 and len(parts[0]) == 1:
            char = parts[0] # Fallback if model outputs FEN chars directly

        if char:
            c, r = int(x // sq_w), int(y // sq_h)
            if 0 <= r < 8 and 0 <= c < 8:
                board[r][c] = char

    return board


def sanitize_board(board):
    """Filters out false positives based on basic chess rules."""
    for r in [0, 7]:
        for c in range(8):
            if board[r][c].lower() == 'p':
                print(f"[DEBUG] Dropped invalid pawn detection at row {r}, col {c}")
                board[r][c] = ""
    return board


def board_dict_to_fen(board_state):
    """Converts board coordinate dictionary to FEN string."""
    rows = []
    for r in range(8):
        empty = 0
        row_str = ""
        for c in range(8):
            if (r, c) in board_state:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += board_state[(r, c)]
            else:
                empty += 1
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)
    
    return "/".join(rows) + " w - - 0 1"


def render_piece(screen, char, cx, cy, images, font):
    """Draws piece sprite or fallback text on pygame surface."""
    if char in images:
        rect = images[char].get_rect(center=(cx, cy))
        screen.blit(images[char], rect)
    else:
        color_fill = (255, 255, 255) if char.isupper() else (20, 20, 20)
        color_stroke = (0, 0, 0) if char.isupper() else (200, 200, 200)
        text_color = (0, 0, 0) if char.isupper() else (255, 255, 255)

        pygame.draw.circle(screen, color_fill, (cx, cy), 25)
        pygame.draw.circle(screen, color_stroke, (cx, cy), 25, 2)
        text = font.render(char.upper() if not char.isupper() else char, True, text_color)
        screen.blit(text, text.get_rect(center=(cx, cy)))


def launch_correction_ui(board_matrix):
    """Pygame interface to manually fix YOLO detection errors."""
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("Board State Verification")
    
    SQ_SIZE, OFS_X, OFS_Y = 70, 20, 20
    font = pygame.font.SysFont('arial', 30, bold=True)
    msg_font = pygame.font.SysFont('arial', 20, bold=True)

    # Preload assets
    sprites = {}
    for char, filename in PIECE_FILES.items():
        path = os.path.join("pieces", filename)
        if os.path.exists(path):
            try:
                img = pygame.image.load(path)
                sprites[char] = pygame.transform.smoothscale(img, (60, 60))
            except pygame.error:
                continue

    # Setup board state
    board_state = {(r, c): board_matrix[r][c] for r in range(8) for c in range(8) if board_matrix[r][c]}
    
    # Palette coordinates
    palette = [
        ('K', 650, 80), ('Q', 730, 80), ('R', 810, 80),
        ('B', 650, 160), ('N', 730, 160), ('P', 810, 160),
        ('k', 650, 260), ('q', 730, 260), ('r', 810, 260),
        ('b', 650, 340), ('n', 730, 340), ('p', 810, 340)
    ]
    
    btn_rect = pygame.Rect(630, 480, 220, 60)
    dragging, dragged_piece = False, None
    err_msg, final_fen = "", None
    running = True

    while running:
        screen.fill((50, 50, 55))
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_rect.collidepoint(mx, my):
                    fen = board_dict_to_fen(board_state)
                    if "K" in fen and "k" in fen:
                        final_fen = fen
                        running = False
                    else:
                        err_msg = "Validation Error: Both kings required."
                else:
                    c, r = (mx - OFS_X) // SQ_SIZE, (my - OFS_Y) // SQ_SIZE
                    if 0 <= r < 8 and 0 <= c < 8 and (r, c) in board_state:
                        dragged_piece = board_state.pop((r, c))
                        dragging, err_msg = True, ""
                    else:
                        for p, px, py in palette:
                            if px - 30 <= mx <= px + 30 and py - 30 <= my <= py + 30:
                                dragged_piece = p
                                dragging, err_msg = True, ""
                                break

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
                c, r = (mx - OFS_X) // SQ_SIZE, (my - OFS_Y) // SQ_SIZE
                if 0 <= r < 8 and 0 <= c < 8:
                    board_state[(r, c)] = dragged_piece
                dragging, dragged_piece = False, None

        # Draw board grid
        for r in range(8):
            for c in range(8):
                color = (235, 236, 208) if (r + c) % 2 == 0 else (115, 149, 82)
                pygame.draw.rect(screen, color, (OFS_X + c * SQ_SIZE, OFS_Y + r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        # Draw board pieces
        for (r, c), piece in board_state.items():
            render_piece(screen, piece, OFS_X + c * SQ_SIZE + 35, OFS_Y + r * SQ_SIZE + 35, sprites, font)

        # Draw UI
        pygame.draw.rect(screen, (70, 70, 75), (610, 20, 260, 420), border_radius=15)
        for p, px, py in palette:
            render_piece(screen, p, px, py, sprites, font)

        pygame.draw.rect(screen, (0, 180, 80), btn_rect, border_radius=10)
        screen.blit(font.render("ANALYZE", True, (255, 255, 255)), (btn_rect.x + 45, btn_rect.y + 12))

        if err_msg:
            screen.blit(msg_font.render(err_msg, True, (255, 50, 50)), (610, 445))

        if dragging and dragged_piece:
            render_piece(screen, dragged_piece, mx, my, sprites, font)

        pygame.display.flip()

    pygame.quit()
    return final_fen


def evaluate_position(engine, fen):
    """Gets the best move from Stockfish."""
    if engine.is_fen_valid(fen):
        engine.set_fen_position(fen)
        return engine.get_best_move()
    return None


def execute_pipeline(frame, yolo_model, stockfish_engine):
    """Runs the full inference and analysis sequence."""
    print("\n[INFO] Starting analysis pipeline...")
    results = yolo_model(frame, verbose=False)
    
    board_matrix = parse_yolo_predictions(results, frame.shape)
    board_matrix = sanitize_board(board_matrix)
    
    final_fen = launch_correction_ui(board_matrix)
    if not final_fen:
        print("[INFO] Analysis aborted by user.")
        return

    print(f"[INFO] Analyzed FEN: {final_fen}")
    best_move = evaluate_position(stockfish_engine, final_fen)
    
    if best_move:
        print(f"\n=> BEST MOVE: {best_move.upper()}\n")
    else:
        print("\n=> ERROR: Engine rejected the FEN position.\n")


def main():
    global board_points
    fetch_assets()
    yolo, engine = init_models()

    cap = cv2.VideoCapture(CAMERA_URL)
    cv2.namedWindow("Chess Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Chess Vision", 800, 600)
    cv2.setMouseCallback("Chess Vision", mouse_click_handler)

    print("\n[CONTROLS]")
    print("[A] Auto-detect board")
    print("[R] Reset points")
    print("[C] Capture & Analyze")
    print("[Q] Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        render_frame = frame.copy()

        # Draw points and polygon
        for pt in board_points:
            cv2.circle(render_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        
        if len(board_points) == 4:
            pts_array = np.array(board_points, dtype=np.int32)
            cv2.polylines(render_frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Chess Vision", render_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            board_points.clear()
            print("[INFO] Selection reset.")
        elif key == ord('a'):
            corners = detect_board_contours(frame)
            if corners is not None:
                board_points = corners.tolist()
                print("[INFO] Board detected automatically.")
            else:
                print("[WARNING] Could not detect board edges.")
        elif key == ord('c'):
            if len(board_points) == 4:
                warped = apply_perspective_transform(frame, board_points)
                cv2.imshow("Crop Preview", warped)
                execute_pipeline(warped, yolo, engine)
            else:
                print("[WARNING] Select exactly 4 corners before analyzing.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
