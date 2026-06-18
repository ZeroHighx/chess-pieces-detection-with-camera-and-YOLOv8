import cv2
import numpy as np
from ultralytics import YOLO
from stockfish import Stockfish
import pygame
import sys
import os
import urllib.request

# ==========================================
# 1. SETTINGS AND PATHS
# ==========================================
# Update this path to where Stockfish is located on your local machine

CAMERA_ADDRESS = "http://YOUR_PHONE_IP:8080/video" # # Use 0 for default webcam, or replace with your IP webcam address
MODEL_PATH = "chess-model-yolov8m.pt" 
STOCKFISH_PATH = r"C:\path\to\your\stockfish\stockfish-windows-x86-64-avx2.exe" # Add your local Stockfish executable path here

print("Sistem yükleniyor, lütfen bekleyin...")
try:
    yolo_model = YOLO(MODEL_PATH)
    stockfish = Stockfish(path=STOCKFISH_PATH)
    print("Modeller başarıyla yüklendi!\n")
except Exception as e:
    print(f"HATA: Modeller yüklenemedi! Ayrıntı: {e}")
    sys.exit()

points = []

# ==========================================
# 🎨 ASSET DOWNLOAD ENGINE
# ==========================================
FEN_TO_URL = {
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

FEN_TO_FILE = {
    'r': 'b_r.png', 'n': 'b_n.png', 'b': 'b_b.png', 'q': 'b_q.png', 'k': 'b_k.png', 'p': 'b_p.png',
    'R': 'w_R.png', 'N': 'w_N.png', 'B': 'w_B.png', 'Q': 'w_Q.png', 'K': 'w_K.png', 'P': 'w_P.png'
}

def download_chess_pieces():
    asset_dir = "pieces"
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
        
    missing_files = False
    for filename in FEN_TO_FILE.values():
        if not os.path.exists(os.path.join(asset_dir, filename)):
            missing_files = True
            break

    if missing_files:
        print("Eksik görseller tespit edildi. Chess.com sunucularından taşlar indiriliyor...")
        for fen_char, url in FEN_TO_URL.items():
            filepath = os.path.join(asset_dir, FEN_TO_FILE[fen_char])
            if not os.path.exists(filepath): 
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"Uyarı: {fen_char} taşı indirilemedi. ({e})")
        print("Görseller başarıyla yüklendi!\n")

download_chess_pieces()

# ==========================================
# 🧠 COMPUTER VISION FUNCTIONS
# ==========================================
def click_event(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])

def auto_detect_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 10000: return None

    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

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

def warp_board(frame, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

# ==========================================
# 🛠️ MATRIX, RULE FILTER AND PYGAME INTERFACE
# ===========================================
def generate_matrix_from_yolo(results, warped_img):
    board_matrix = [["" for _ in range(8)] for _ in range(8)]
    h, w = warped_img.shape[:2]
    sq_h, sq_w = h / 8, w / 8
    
    FEN_MAP = {
        'black-rook': 'r', 'black_rook': 'r', 'r': 'r', 'black rook': 'r',
        'black-knight': 'n', 'black_knight': 'n', 'n': 'n', 'black knight': 'n',
        'black-bishop': 'b', 'black_bishop': 'b', 'b': 'b', 'black bishop': 'b',
        'black-queen': 'q', 'black_queen': 'q', 'q': 'q', 'black queen': 'q',
        'black-king': 'k', 'black_king': 'k', 'k': 'k', 'black king': 'k',
        'black-pawn': 'p', 'black_pawn': 'p', 'p': 'p', 'black pawn': 'p',
        'white-rook': 'R', 'white_rook': 'R', 'R': 'R', 'white rook': 'R',
        'white-knight': 'N', 'white_knight': 'N', 'N': 'N', 'white knight': 'N',
        'white-bishop': 'B', 'white_bishop': 'B', 'B': 'B', 'white bishop': 'B',
        'white-queen': 'Q', 'white_queen': 'Q', 'Q': 'Q', 'white queen': 'Q',
        'white-king': 'K', 'white_king': 'K', 'K': 'K', 'white king': 'K',
        'white-pawn': 'P', 'white_pawn': 'P', 'P': 'P', 'white pawn': 'P'
    }
    
    for box in results[0].boxes:
        x_center, y_center = int(box.xywh[0][0]), int(box.xywh[0][1])
        class_id = int(box.cls[0])
        raw_name = results[0].names[class_id].lower()
        piece_char = FEN_MAP.get(raw_name, "")
        
        if piece_char:
            col, row = int(x_center // sq_w), int(y_center // sq_h)
            if 0 <= row < 8 and 0 <= col < 8:
                board_matrix[row][col] = piece_char
    return board_matrix

def apply_chess_rules(board_matrix):
    """Satranç kurallarına uymayan yapay zeka halüsinasyonlarını otomatik temizler."""
    for r in range(8):
        for c in range(8):
            piece = board_matrix[r][c]
            if not piece: continue
            
            # KURAL: 1. (r=7) ve 8. (r=0) yatayda piyon olamaz!
            if piece.lower() == 'p' and (r == 0 or r == 7):
                print(f"Hakem Filtresi: Yapay zeka {r}. indeks satırında hatalı bir piyon buldu. Taş otomatik olarak silindi!")
                board_matrix[r][c] = ""
                
    return board_matrix

def generate_fen_from_dict(board_dict):
    rows = []
    for r in range(8):
        empty = 0
        row_str = ""
        for c in range(8):
            if (r, c) in board_dict:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += board_dict[(r, c)]
            else:
                empty += 1
        if empty > 0: row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows) + " w - - 0 1"

def draw_piece(screen, piece_char, center_x, center_y, loaded_images, font):
    if piece_char in loaded_images:
        img = loaded_images[piece_char]
        rect = img.get_rect(center=(center_x, center_y))
        screen.blit(img, rect)
    else:
        if piece_char.isupper():
            pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 25)
            pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), 25, 2)
            text = font.render(piece_char, True, (0, 0, 0))
        else:
            pygame.draw.circle(screen, (20, 20, 20), (center_x, center_y), 25)
            pygame.draw.circle(screen, (200, 200, 200), (center_x, center_y), 25, 2)
            text = font.render(piece_char.upper(), True, (255, 255, 255))
        text_rect = text.get_rect(center=(center_x, center_y))
        screen.blit(text, text_rect)

def edit_board_pygame(board_matrix):
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("Yapay Zeka Hata Ayıklayıcı (Drag & Drop)")
    
    SQ_SIZE = 70
    OFS_X, OFS_Y = 20, 20
    font = pygame.font.SysFont('arial', 30, bold=True)
    msg_font = pygame.font.SysFont('arial', 20, bold=True)

    loaded_images = {}
    for fen_char, filename in FEN_TO_FILE.items():
        filepath = os.path.join("pieces", filename)
        if os.path.exists(filepath):
            try:
                img = pygame.image.load(filepath)
                img = pygame.transform.smoothscale(img, (60, 60)) 
                loaded_images[fen_char] = img
            except: pass

    board_state = {}
    for r in range(8):
        for c in range(8):
            if board_matrix[r][c] != "":
                board_state[(r, c)] = board_matrix[r][c]

    side_pieces = [
        ('K', 650, 80), ('Q', 730, 80), ('R', 810, 80),
        ('B', 650, 160), ('N', 730, 160), ('P', 810, 160),
        ('k', 650, 260), ('q', 730, 260), ('r', 810, 260),
        ('b', 650, 340), ('n', 730, 340), ('p', 810, 340)
    ]

    btn_rect = pygame.Rect(630, 480, 220, 60)
    dragging = False
    dragged_piece = None
    error_msg = ""
    
    running = True
    final_fen = None

    while running:
        screen.fill((50, 50, 55)) 
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if btn_rect.collidepoint(mouse_x, mouse_y):
                        fen_attempt = generate_fen_from_dict(board_state)
                        if "K" in fen_attempt and "k" in fen_attempt:
                            final_fen = fen_attempt
                            running = False
                        else:
                            error_msg = "HATA: Her iki Şah (K/k) tahtada olmali!"
                    else:
                        c, r = (mouse_x - OFS_X) // SQ_SIZE, (mouse_y - OFS_Y) // SQ_SIZE
                        if 0 <= r < 8 and 0 <= c < 8:
                            if (r, c) in board_state:
                                dragged_piece = board_state.pop((r, c))
                                dragging = True
                                error_msg = ""
                        else:
                            for p, px, py in side_pieces:
                                if (px-30) <= mouse_x <= (px+30) and (py-30) <= mouse_y <= (py+30):
                                    dragged_piece = p
                                    dragging = True
                                    error_msg = ""
                                    break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    c, r = (mouse_x - OFS_X) // SQ_SIZE, (mouse_y - OFS_Y) // SQ_SIZE
                    if 0 <= r < 8 and 0 <= c < 8:
                        board_state[(r, c)] = dragged_piece
                    dragging = False
                    dragged_piece = None

        for r in range(8):
            for c in range(8):
                color = (235, 236, 208) if (r+c)%2==0 else (115, 149, 82) 
                pygame.draw.rect(screen, color, (OFS_X + c*SQ_SIZE, OFS_Y + r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

        for (r, c), piece in board_state.items():
            draw_piece(screen, piece, OFS_X + c*SQ_SIZE + 35, OFS_Y + r*SQ_SIZE + 35, loaded_images, font)

        pygame.draw.rect(screen, (70, 70, 75), (610, 20, 260, 420), border_radius=15)
        
        for p, px, py in side_pieces:
            draw_piece(screen, p, px, py, loaded_images, font)

        pygame.draw.rect(screen, (0, 180, 80), btn_rect, border_radius=10)
        btn_text = font.render("ANALİZE GÖNDER", True, (255, 255, 255))
        screen.blit(btn_text, (btn_rect.x + 15, btn_rect.y + 12))

        if error_msg:
            err_text = msg_font.render(error_msg, True, (255, 50, 50))
            screen.blit(err_text, (610, 445))

        if dragging and dragged_piece:
            draw_piece(screen, dragged_piece, mouse_x, mouse_y, loaded_images, font)

        pygame.display.flip()

    pygame.quit()
    return final_fen

# ==========================================
# 🤖 ANALYSIS PIPELINE
# ===========================================
def analyze_with_stockfish(fen_code):
    try:
        if stockfish.is_fen_valid(fen_code):
            stockfish.set_fen_position(fen_code)
            return stockfish.get_best_move()
        else:
            return "Geçersiz FEN formati!"
    except Exception as e:
        return f"Motor Hatası: {e}"

def run_pipeline(warped_img):
    print("\n--- ANALİZ BAŞLADI ---")
    results = yolo_model(warped_img, verbose=False)
    
    print("1. Yerel YOLO Matrisi oluşturuldu.")
    board_matrix = generate_matrix_from_yolo(results, warped_img)
    
    print("2. Hakem Filtresi uygulanıyor...")
    board_matrix = apply_chess_rules(board_matrix)
    
    print("3. Pygame arayüzü açılıyor...")
    validated_fen = edit_board_pygame(board_matrix)
    
    if not validated_fen:
        print("Arayüz kapatıldı, analiz iptal edildi.\n")
        return
        
    print(f"4. Doğrulanmış FEN: {validated_fen}")
    print("5. Stockfish hamle hesaplıyor...")
    
    best_move = analyze_with_stockfish(validated_fen)
    
    print(f"\n=================================")
    print(f"♟️ ÖNERİLEN EN İYİ HAMLE: {str(best_move).upper()}")
    print(f"=================================\n")

# ==========================================
# 🎥 MAIN CAMERA LOOP
# ===========================================
def main():
    global points
    cap = cv2.VideoCapture(CAMERA_ADDRESS)
    
    cv2.namedWindow("Real-Time Chess Vision", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Chess Vision", 800, 600)
    cv2.setMouseCallback("Real-Time Chess Vision", click_event)

    print("Kamera Başlatıldı! Kısayollar:\n"
          "[A] Otomatik Tahta Tespiti\n"
          "[R] Noktaları Sıfırla\n"
          "[C] Analizi Başlat (Sürükle-Bırak Arayüzlü)\n"
          "[Q] Çıkış\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        if len(points) > 0:
            for pt in points:
                cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            if len(points) == 4:
                cv2.polylines(display_frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow("Real-Time Chess Vision", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            print("Noktalar sıfırlandı.")
        elif key == ord('a'):
            detected_corners = auto_detect_board(frame)
            if detected_corners is not None:
                points = detected_corners.tolist()
                print("Başarılı! Tahta köşeleri otomatik bulundu.")
            else:
                print("HATA: Tahta net bulunamadı.")
        elif key == ord('c'):
            if len(points) == 4:
                warped = warp_board(frame, points)
                cv2.imshow("Warped Board", warped)
                run_pipeline(warped)
            else:
                print("Analiz için tam 4 köşe gerekli!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
