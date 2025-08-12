
import webbrowser
import threading
import os, json, csv, re, io, zipfile
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, send_from_directory
from flask import request, redirect, url_for, jsonify
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
from bs4 import BeautifulSoup
import cv2
import sys
import numpy as np
from io import BytesIO
import mediapipe as mp
from rembg import remove
from pathlib import Path
from math import ceil
import base64
import shutil
from werkzeug.utils import secure_filename
import glob
import itertools 
from flask import after_this_request
from werkzeug.utils import secure_filename


app = Flask(__name__)

os.environ["NUMBA_DISABLE_CACHE"] = "1"
import os
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.getcwd(), "numba_cache")

Image.MAX_IMAGE_PIXELS = None
mp_face_mesh = mp.solutions.face_mesh

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    WORK_DIR = os.path.dirname(sys.executable)  # location where .exe runs
else:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    WORK_DIR = base_dir

counter = 1
# 🟢 STATIC & TEMPLATES for reading only
# Working directory for writing files
TEMPLATE_FOLDER = os.path.join(base_dir, "templates")
STATIC_FOLDER = os.path.join(base_dir, "static")
LAYOUT_JSON_PATH = os.path.join(STATIC_FOLDER, "layout.json")
from flask import Flask
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
CAPTURE_FOLDER = os.path.join(base_dir, 'static', 'captured')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")
CLEANED_FOLDER = os.path.join(WORK_DIR, "cleaned")
CAPTURE_DIR = os.path.join(WORK_DIR, "captured")
PHOTO_FOLDER = os.path.join(WORK_DIR, "photos")
CSV_FILE = os.path.join(STATIC_FOLDER, "students.csv")
MASK_PATH = os.path.join(UPLOAD_FOLDER, "mask.png")
ZIP_PATH = os.path.join(base_dir, 'captured_photos.zip')
CLEANED_FOLDER = "static/cleaned"
# Ensure all folders exist
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(PHOTO_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(CAPTURE_FOLDER, exist_ok=True)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


counter = itertools.count(1)  

try:
    with open(LAYOUT_JSON_PATH, 'r') as f:
        layout = json.load(f)
except:
    layout = {}
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


def safe_filename(name):
    name = os.path.splitext(name)[0]
    name = re.sub(r'[^\w\-_.]', '_', name)
    return name



@app.route("/")
def launcher():
    return render_template("launcher.html")

@app.route("/download-guide")
def download_guide():
    return send_from_directory(
        directory=os.path.join(app.static_folder),
        path="user_guide.pdf",
        as_attachment=True
        
    )



@app.route("/layout")
def layout_editor():
    fields = []
    if os.path.exists("layout.json"):
        with open("layout.json", "r") as file:
            layout = json.load(file)
            for f in layout.get("fields", []):
                soup = BeautifulSoup(f.get("label", ""), "html.parser")
                label = soup.get_text()
                fields.append((label, f["x"], f["y"], f["w"], f["h"]))
    return render_template("layout.html", fields=fields)

# 🔹 Background Remover Page
@app.route("/clean")
def clean_ui():
    return render_template("clean.html")
@app.route("/camera")
def camera():
    return render_template("camera.html")




@app.route("/save-photo", methods=["POST"])
def save_photo():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "No image data"}), 400

    # Decode image
    header, encoded = data.split(",", 1)
    img_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(img_data))

    # Find next index
    existing_files = sorted([
        f for f in os.listdir(CAPTURE_FOLDER) if f.endswith(".jpg")
    ])
    next_index = len(existing_files) + 1
    filename = f"{next_index}.jpg"
    path = os.path.join(CAPTURE_FOLDER, filename)

    # Save image
    image.save(path, "JPEG")
    return jsonify({"filename": f"captured/{filename}"}), 200

# Delete photo route
@app.route("/delete-photo")
def delete_photo():
    filename = request.args.get("filename")
    if not filename:
        return "Missing filename", 400

    path = os.path.join(CAPTURE_FOLDER, os.path.basename(filename))
    if os.path.exists(path):
        os.remove(path)
    return "Deleted", 200

# Download all as ZIP
@app.route("/download-photos")
def download_photos():
    # Remove old ZIP if exists
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    # Rename files in temp folder starting from 1.jpg
    temp_dir = os.path.join(base_dir, 'temp_rename')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(CAPTURE_FOLDER) if f.endswith(".jpg")
    ])

    for i, img_name in enumerate(images, start=1):
        src = os.path.join(CAPTURE_FOLDER, img_name)
        dst = os.path.join(temp_dir, f"{i}.jpg")
        shutil.copy2(src, dst)

    # Create ZIP
    with zipfile.ZipFile(ZIP_PATH, "w") as zipf:
        for fname in sorted(os.listdir(temp_dir), key=lambda x: int(x.split(".")[0])):
            fpath = os.path.join(temp_dir, fname)
            zipf.write(fpath, fname)

    shutil.rmtree(temp_dir)
    return send_file(ZIP_PATH, as_attachment=True)

# Serve captured images
@app.route('/captured/<filename>')
def captured_file(filename):
    return send_from_directory(CAPTURE_FOLDER, filename)


@app.route("/process", methods=["POST"])
def process():
    # Delete all old files before processing
    for old_file in os.listdir(CLEANED_FOLDER):
        if old_file.endswith(".jpg") or old_file.endswith(".png"):
            try:
                os.remove(os.path.join(CLEANED_FOLDER, old_file))
                print(f"🗑 Deleted old file: {old_file}")
            except Exception as e:
                print(f"❌ Could not delete old file: {e}")

    processed = []
    for file in request.files.getlist("photos"):
        safe_name = safe_filename(file.filename)
        fname = f"{safe_name}.png"

        img_data = remove(file.read())
        img = Image.open(BytesIO(img_data)).convert("RGBA")

        save_path = os.path.join(CLEANED_FOLDER, fname)
        img.save(save_path, format="PNG")

        processed.append(fname)

    return jsonify({"processed": processed})



# --- Background Remover /recolor
@app.route("/recolor", methods=["POST"])
def recolor():
    fname = request.form["filename"]
    hex_color = request.form["color"].lstrip("#")
    color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    size = (600, 800)

    transparent_path = os.path.join("static/cleaned", fname)
    img = Image.open(transparent_path).convert("RGBA")

    bg_img = Image.new("RGBA", size, color + (255,))
    
    # सीधे original image को size में fit करें, crop न करें
    final_img = ImageOps.contain(img, size, method=Image.LANCZOS)

    # सीधे background पर फिक्स करें
    offset_x = (size[0] - final_img.width) // 2
    offset_y = (size[1] - final_img.height) // 2
    bg_img.paste(final_img, (offset_x, offset_y), final_img)

    jpg_name = fname.replace(".png", ".jpg")
    jpg_path = os.path.join("static/cleaned", jpg_name)
    bg_img.convert("RGB").save(jpg_path, format="JPEG", quality=95)
    return jsonify({"updated": jpg_name})

def clean_old_files():
    """Deletes all old files before processing new ones."""
    for fname in os.listdir(CLEANED_FOLDER):
        path = os.path.join(CLEANED_FOLDER, fname)
        try:
            os.remove(path)
            print(f"🗑 Deleted old file: {path}")
        except Exception as e:
            print(f"❌ Error deleting {path}: {e}")


@app.route("/download/<filename>")
def download_one(filename):
    jpg_path = os.path.join(CLEANED_FOLDER, filename)

    if not os.path.exists(jpg_path):
        return "File not found", 404

    with open(jpg_path, "rb") as f:
        data = f.read()

    response = send_file(
        BytesIO(data),
        as_attachment=True,
        download_name=filename,
        mimetype="image/jpeg"
    )

    # ✅ Delete after sending
    @response.call_on_close
    def cleanup():
        try:
            os.remove(jpg_path)
            print(f"✅ Deleted: {jpg_path}")
        except Exception as e:
            print(f"❌ Could not delete {jpg_path}: {e}")

    return response


@app.route("/download-all")
def download_all():
    zip_buffer = BytesIO()
    zip_files = [f for f in os.listdir(CLEANED_FOLDER) if f.endswith(".jpg")]

    if not zip_files:
        return "No files to download", 404

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in zip_files:
            file_path = os.path.join(CLEANED_FOLDER, fname)
            zf.write(file_path, arcname=fname)

    zip_buffer.seek(0)

    response = send_file(
        zip_buffer,
        mimetype="application/zip",
        download_name="cleaned_images.zip",
        as_attachment=True
    )

    # ✅ Delete all files after sending ZIP
    @response.call_on_close
    def cleanup():
        for fname in zip_files:
            path = os.path.join(CLEANED_FOLDER, fname)
            try:
                os.remove(path)
                print(f"✅ Deleted: {path}")
            except Exception as e:
                print(f"❌ Could not delete {path}: {e}")

    return response




def parse_color(color_hex):
    color_hex = color_hex.lstrip("#")
    if len(color_hex) == 6:
        return tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    return (0, 0, 0)

def create_gradient_border(size, radius, thickness, color1, color2):
    w, h = size
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (w, h)], radius=radius, fill=255)
    gradient = Image.new("RGBA", (w, h), color1)
    for y in range(h):
        ratio = y / h
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        ImageDraw.Draw(gradient).line([(0, y), (w, y)], fill=(r, g, b), width=1)
    base.paste(gradient, (0, 0), mask)
    inner = Image.new("L", (w - 2 * thickness, h - 2 * thickness), 0)
    ImageDraw.Draw(inner).rounded_rectangle([(0, 0), (w - 2 * thickness - 1, h - 2 * thickness - 1)],
                                            radius=max(0, radius - thickness), fill=255)
    mask_inner = Image.new("L", (w, h), 0)
    mask_inner.paste(inner, (thickness, thickness))
    final_mask = ImageChops.subtract(mask, mask_inner)
    return Image.composite(base, Image.new("RGBA", (w, h), (0, 0, 0, 0)), final_mask)

def create_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, folder_path)
                zipf.write(filepath, arcname)
    if os.path.exists(zip_path):
        os.remove(zip_path)



@app.route("/upload-template", methods=["POST"])
def upload_template():
    file = request.files.get("template")
    if file:
        upload_path = os.path.join(app.static_folder, "uploads")
        os.makedirs(upload_path, exist_ok=True)
        file.save(os.path.join(upload_path, "template.jpg"))
        return "✅ Template uploaded"
    return "❌ No file received", 400

@app.route("/upload-mask", methods=["POST"])
def upload_mask():
    file = request.files["mask"]
    if file.filename.lower().endswith((".jpg", ".jpeg")):
        img = Image.open(file).convert("L")
        img.save(MASK_PATH)
    else:
        img = Image.open(file).convert("RGBA")
        img.split()[-1].save(MASK_PATH)
    return "✅ Mask uploaded"

@app.route("/save-layout", methods=["POST"])
def save_layout():
    data = request.get_json()
    print("📦 Layout Data:", data)  # ✅ Debug
    path = os.path.join("static", "layout.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return "✅ Layout saved"







@app.route("/upload-csv", methods=["POST"])
def upload_csv():
    file = request.files.get("csv")
    if file and file.filename.lower().endswith(".csv"):
        try:
            os.makedirs(STATIC_FOLDER, exist_ok=True)  # make sure folder exists
            file.save(CSV_FILE)
            return jsonify({"message": "✅ CSV uploaded to static/"}), 200
        except Exception as e:
            return jsonify({"error": f"❌ Failed to save CSV: {str(e)}"}), 500
    return jsonify({"error": "❌ Invalid CSV file"}), 400



@app.route("/delete-csv", methods=["POST"])
def delete_csv():
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        return "✅ CSV deleted", 200
    return "⚠️ No CSV file found", 404

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')



@app.route("/upload-photos", methods=["POST"])
def upload_photos():
    files = request.files.getlist("photos[]")

    for file in files:
        filename = file.filename.replace("\\", "/")  # Windows fix

        # ✅ Remove first "photos/" from filename if present
        if filename.startswith("photos/"):
            filename = filename[len("photos/"):]

        # ✅ Now create correct path inside photos/
        save_path = os.path.join("static", "photos", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

    return "✅ All photos uploaded successfully"


@app.route("/delete-photos-all", methods=["POST"])
def delete_photos_all():
    photo_folder = os.path.join(app.root_path, "static", "photos")
    print("Deleting:", photo_folder)

    if os.path.exists(photo_folder):
        shutil.rmtree(photo_folder)
    os.makedirs(photo_folder)
    return "✅ All photos and folders deleted"

@app.route("/photo/<path:filename>")
def get_photo(filename):
    photo_path = os.path.join(PHOTO_FOLDER, filename)
    if not os.path.exists(photo_path):
        return "❌ Photo file nahi mila!"
    return send_file(photo_path)

@app.route("/students")
def list_students():
    if not os.path.exists(CSV_FILE):
        return "❌ CSV file static folder me nahi mila!"
    students = list(csv.DictReader(open(CSV_FILE)))
    stu = students[0]

    return render_template("students.html", students=students)

@app.route("/fonts/<path:filename>")
def serve_font(filename):
    return send_from_directory("static/fonts", filename)  # ✅ fixed path

@app.route("/fonts")
def list_fonts():
    return jsonify([
        { "file": "arial.ttf", "name": "Arial" },
        { "file": "arialbd.ttf", "name": "Arial Bold" },
        { "file": "arialbi.ttf", "name": "Arial Bold Italic" },
        { "file": "ariali.ttf", "name": "Arial Italic" },
        { "file": "ariblk.ttf", "name": "Arial Black" },
        { "file": "calibri.ttf", "name": "Calibri" },
        { "file": "calibrib.ttf", "name": "Calibri Bold" },
        { "file": "calibrii.ttf", "name": "Calibri Italic" },
        { "file": "calibril.ttf", "name": "Calibri Light" },
        { "file": "calibrili.ttf", "name": "Calibri Light Italic" },
        { "file": "calibriz.ttf", "name": "Calibri Bold Italic" },
        { "file": "times.ttf", "name": "Times New Roman" },
        { "file": "timesbd.ttf", "name": "Times Bold" },
        { "file": "timesbi.ttf", "name": "Times Bold Italic" },
        { "file": "timesi.ttf", "name": "Times Italic" }
    ])

@app.route("/generate-sample")
def generate_sample():
    tpl_path = os.path.join(UPLOAD_FOLDER, "template.jpg")
    if not os.path.exists(tpl_path):
        return "❌ Template not found"

    if not os.path.exists(CSV_FILE):
        return "❌ CSV file not found"

    layout_path = os.path.join(base_dir, "static", "layout.json")
    with open(layout_path, 'r') as f:
        layout = json.load(f)

    students = list(csv.DictReader(open(CSV_FILE)))

    if not students:
        return "❌ CSV is empty"

    stu = students[0]  # ✅ THIS FIXES YOUR ERROR

    tpl = Image.open(tpl_path).convert("RGBA")
    card = tpl.copy()
    draw = ImageDraw.Draw(card)

    for fld in layout["fields"]:
        label_text = BeautifulSoup(fld["label"], "html.parser").get_text().strip()
        label_key = label_text.lower()
        x, y, w, h = int(fld["x"]), int(fld["y"]), int(fld["w"]), int(fld["h"])
        font_size = int(fld.get("size", 24)) * 4.11
        font_color = parse_color(fld.get("color", "#000000"))
        align = fld.get("align", "left")
        font_file = fld.get("font", "arial.ttf")
        font_path = os.path.join("static", "fonts", font_file)

        if not os.path.exists(font_path):
            font_path = os.path.join("static", "fonts", "arial.ttf")

        font = ImageFont.truetype(font_path, font_size)

        if label_key in ["photo", "student photo"]:
            photo_id = stu.get("Roll No", "").strip()
            photo_path = os.path.join("static", "photos", f"{photo_id}.jpg")

            bg_layer = Image.new("RGBA", (w, h), (255, 255, 255, 0))

            if os.path.exists(photo_path):
                try:
                    pil_image = Image.open(photo_path).convert("RGB")
                    img = np.array(pil_image)

                    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                    if results.multi_face_landmarks:
                        h_img, w_img = img.shape[:2]
                        landmarks = results.multi_face_landmarks[0].landmark
                        xs = [lm.x * w_img for lm in landmarks]
                        ys = [lm.y * h_img for lm in landmarks]
                        min_x, max_x = int(min(xs)), int(max(xs))
                        min_y, max_y = int(min(ys)), int(max(ys))

                        face_h = max_y - min_y
                        cx = (min_x + max_x) // 2
                        cy = (min_y + max_y) // 2

                        crop_h = int(face_h * 2.5)
                        crop_w = int(crop_h * (w / h))
                        offset_y = int(crop_h * 0.1)

                        cx1 = max(0, cx - crop_w // 2)
                        cy1 = max(0, cy - crop_h // 2 - offset_y)
                        cx2 = min(w_img, cx + crop_w // 2)
                        cy2 = min(h_img, cy + crop_h // 2 - offset_y)

                        cropped_cv = img[cy1:cy2, cx1:cx2]
                        cropped_pil = Image.fromarray(cropped_cv)
                    else:
                        cropped_pil = pil_image.copy()

                except Exception as e:
                    print("❌ Photo processing error:", e)
                    cropped_pil = pil_image.copy()

                # Resize and paste
                final_crop = ImageOps.fit(cropped_pil, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))
                offset_x = (w - final_crop.width) // 1
                offset_y = (h - final_crop.height) // 1
                bg_layer.paste(final_crop, (offset_x, offset_y))

                # Apply mask if available
                if os.path.exists(MASK_PATH):
                    try:
                        mask = Image.open(MASK_PATH).resize((w, h)).convert("L")
                        bg_layer.putalpha(mask)
                    except Exception as e:
                        print("❌ Mask apply error:", e)

                # Gradient border
                try:
                    radius = int(fld.get("radius", min(w, h) // 6))
                    thickness = int(fld.get("thickness", 6))
                    c1 = parse_color(fld.get("color1", "#ff0000"))
                    c2 = parse_color(fld.get("color2", "#0000ff"))
                except Exception as e:
                    print("❌ Border parse error:", e)
                    radius, thickness = min(w, h) // 6, 6
                    c1, c2 = (255, 0, 0), (0, 0, 255)

                gradient = create_gradient_border((w, h), radius, thickness, c1, c2)
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
                bg_layer.putalpha(mask)
                bg_layer = Image.alpha_composite(bg_layer, gradient)

                # Paste final
                paste = Image.new("RGBA", card.size)
                paste.paste(bg_layer, (x, y), bg_layer)
                card.alpha_composite(paste)


        else:
            value = next((stu[key] for key in stu if key.lower() == label_key), "")
            if value:
                try:
                    if not os.path.exists(font_path):
                        font = ImageFont.load_default()
                    else:
                        font = ImageFont.truetype(font_path, font_size)
                    draw_multiline_text(draw, value, font, font_color, x, y, w, h, align)
                except Exception as e:
                    print("❌ Text draw error:", e)

            elif label_key in ["signature", "barcode", "qrcode"]:
                folder_map = {
                     "signature": os.path.join("static", "photos", "signature"),
                     "barcode": os.path.join("static", "photos", "barcode"),
                    "qrcode": os.path.join("static", "photos", "qrcode"),
         }
                image_folder = folder_map.get(label_key)
                if not image_folder:
                    continue

                photo_id = stu.get("Roll No", "").strip()
                image_path = os.path.join(image_folder, f"{photo_id}.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{photo_id}.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{photo_id}.jpeg")

                if os.path.exists(image_path):
                    img = Image.open(image_path).convert("RGBA")
                    img_copy = img.copy()
                    img_copy.thumbnail((w, h), Image.LANCZOS)

                    # Center image in the field box
                    offset_x = x + (w - img_copy.width) // 2
                    offset_y = y + (h - img_copy.height) // 2

                    paste_layer = Image.new("RGBA", card.size)
                    paste_layer.paste(img_copy, (offset_x, offset_y), img_copy)
                    card.alpha_composite(paste_layer)
       
    pdf_bytes = BytesIO()
    card.convert("RGB").save(pdf_bytes, format="PDF", resolution=300)
    pdf_bytes.seek(0)
    generate_cards_and_sheets()
    print("Student:", stu)
    print("Fields:", layout["fields"])

    return send_file(pdf_bytes, mimetype="application/pdf", as_attachment=True, download_name="id_card.pdf")
    
def draw_multiline_text(draw, text, font, fill, x, y, w, h, align="left", line_spacing=4):
    lines = []
    words = text.split()
    while words:
        line = ''
        while words:
            test_line = line + ('' if line == '' else ' ') + words[0]
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= w:
                line = test_line
                words.pop(0)
            else:
                break
        if not line:
            line = words.pop(0)
        lines.append(line)

    y_offset = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw_x = x + (w - (bbox[2] - bbox[0])) / 2 if align == "center" else (x + (w - (bbox[2] - bbox[0])) if align == "right" else x)
        draw.text((draw_x, y_offset), line, font=font, fill=fill)
        y_offset += (bbox[3] - bbox[1]) + line_spacing
      
from math import ceil
from pathlib import Path

from PIL import Image
import math

def generate_cards_and_sheets():
    layout_path = os.path.join(base_dir, "static", "layout.json")
    with open(layout_path, 'r') as f:
            layout = json.load(f)

    students = list(csv.DictReader(open(CSV_FILE)))

    card_px_w = layout["template_size"]["w"]
    card_px_h = layout["template_size"]["h"]
    dpi = 300

    # A4 in pixels
    a4_mm_w, a4_mm_h = 210, 297
    a4_w = int(a4_mm_w * dpi / 25.4)
    a4_h = int(a4_mm_h * dpi / 25.4)

    desktop = str(Path.home() / "Desktop")
    card_dir = os.path.join(desktop, "cards")
    sheet_dir = os.path.join(desktop, "sheets")
    os.makedirs(card_dir, exist_ok=True)
    os.makedirs(sheet_dir, exist_ok=True)

    tpl_path = os.path.join("static/uploads", "template.jpg")
    if not os.path.exists(tpl_path):
        raise FileNotFoundError("❌ Template not found")

    card_paths = []

    for stu in students:
        tpl = Image.open(tpl_path).convert("RGBA")
        card = tpl.copy()
        draw = ImageDraw.Draw(card)

        for fld in layout["fields"]:
            # हर loop के शुरुआत में
            label_text = BeautifulSoup(fld["label"], "html.parser").get_text().strip()
            label_key = label_text.lower()   # इसी लाइक assign करो
            folder = os.path.join(base_dir, "static", "photos", label_key)

            x, y, w, h = int(fld["x"]), int(fld["y"]), int(fld["w"]), int(fld["h"])
            font_size = int(fld.get("size", 24)) * 4.11
            color = parse_color(fld.get("color", "#000000"))
            align = fld.get("align", "left")
            font_file = fld.get("font", "arial.ttf")
            font_path = os.path.join("static", "fonts", font_file)

            if label_key in ["photo", "student photo"]:
                roll = stu.get("Roll No", "").strip()
                # Correct path: static/photos/1.jpg, 2.jpg, ...
                photo_path = os.path.join("static", "photos", f"{roll}.jpg")

                bg = Image.new("RGBA", (w, h), (255, 255, 255, 0))
                cropped_pil = None


                if os.path.exists(photo_path):
                    try:
                        img = np.array(Image.open(photo_path).convert("RGB"))
                        with mp_face_mesh.FaceMesh(static_image_mode=True) as fm:
                            res = fm.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                            if res.multi_face_landmarks:
                                lm = res.multi_face_landmarks[0].landmark
                                h_img, w_img = img.shape[:2]
                                xs = [l.x * w_img for l in lm]
                                ys = [l.y * h_img for l in lm]
                                min_x, max_x = int(min(xs)), int(max(xs))
                                min_y, max_y = int(min(ys)), int(max(ys))
                                face_h = max_y - min_y
                                cx = (min_x + max_x) // 2
                                cy = (min_y + max_y) // 2

                                ch = int(face_h * 2.8)
                                cw = int(ch * (w / h))

                                # ✅ shift crop box vertically to give space above head
                                offset = int(face_h * 0.4)  # adjust this: 0.3–0.5 recommended

                                cy1 = max(0, cy - ch // 2 - offset)
                                cy2 = min(h_img, cy + ch // 2 - offset)

                                cx1 = max(0, cx - cw // 2)
                                cx2 = min(w_img, cx + cw // 2)

                                cropped_cv = img[cy1:cy2, cx1:cx2]

                                cropped_pil = Image.fromarray(cropped_cv)
                    except Exception as e:
                        print("❌ Face error", e)

                    if cropped_pil is None:
                        cropped_pil = Image.open(photo_path)

                    final = ImageOps.fit(cropped_pil, (w, h), method=Image.LANCZOS)
                    bg.paste(final, ((w - final.width) // 2, (h - final.height) // 2))

                    if os.path.exists(MASK_PATH):
                        mask = Image.open(MASK_PATH).resize((w, h)).convert("L")
                        bg.putalpha(mask)

                radius = int(fld.get("radius", min(w, h) // 6))
                thickness = int(fld.get("thickness", 6))
                c1 = parse_color(fld.get("color1", "#ff0000"))
                c2 = parse_color(fld.get("color2", "#0000ff"))
                gradient = create_gradient_border((w, h), radius, thickness, c1, c2)
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
                bg.putalpha(mask)
                bg = Image.alpha_composite(bg, gradient)

                paste = Image.new("RGBA", card.size)
                paste.paste(bg, (x, y), bg)
                card.alpha_composite(paste)

            elif label_key in ["signature", "barcode", "qrcode"]:
                folder_map = {
                     "signature": os.path.join("static", "photos", "signature"),
                     "barcode": os.path.join("static", "photos", "barcode"),
                     "qrcode": os.path.join("static", "photos", "qrcode"),
         }
                image_folder = folder_map.get(label_key)
                if not image_folder:
                    continue

                photo_id = stu.get("Roll No", "").strip()
                image_path = os.path.join(image_folder, f"{photo_id}.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{photo_id}.jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{photo_id}.jpeg")

                if os.path.exists(image_path):
                    img = Image.open(image_path).convert("RGBA")
                    img_copy = img.copy()
                    img_copy.thumbnail((w, h), Image.LANCZOS)

                    # Center image in the field box
                    offset_x = x + (w - img_copy.width) // 2
                    offset_y = y + (h - img_copy.height) // 2

                    paste_layer = Image.new("RGBA", card.size)
                    paste_layer.paste(img_copy, (offset_x, offset_y), img_copy)
                    card.alpha_composite(paste_layer)


            else:
                value = next((stu[k] for k in stu if k.lower() == label_key), "")
                if value:
                    font = ImageFont.truetype(font_path, font_size) if os.path.exists(font_path) else ImageFont.load_default()
                    draw_multiline_text(draw, value, font, color, x, y, w, h, align)

        out_path = os.path.join(card_dir, f"{stu['Roll No']}.jpg")
        card.convert("RGB").save(out_path, dpi=(300, 300), quality=95)
        card_paths.append(out_path)

    # Sheet making
        # Sheet making with center alignment
    if not card_paths:
        return
    gap = 0
    cols = max(1, (a4_w + gap) // (card_px_w + gap))
    rows = max(1, (a4_h + gap) // (card_px_h + gap))
    per_sheet = cols * rows

    chunks = [card_paths[i:i + per_sheet] for i in range(0, len(card_paths), per_sheet)]
    for i, chunk in enumerate(chunks):
        sheet = Image.new("RGB", (a4_w, a4_h), "white")
        total_cards = len(chunk)
        total_rows = math.ceil(total_cards / cols)

        for idx, path in enumerate(chunk):
            img = Image.open(path).resize((card_px_w, card_px_h), Image.LANCZOS)
            row = idx // cols
            col = idx % cols

            # Center horizontally if this row has fewer cards
            cards_in_this_row = min(cols, total_cards - row * cols)
            row_width = cards_in_this_row * card_px_w + (cards_in_this_row - 1) * gap
            start_x = (a4_w - row_width) // 2

            # Center vertically on last sheet
            start_y = (a4_h - (total_rows * card_px_h + (total_rows - 1) * gap)) // 2

            x = start_x + col * (card_px_w + gap)
            y = start_y + row * (card_px_h + gap)

            sheet.paste(img, (x, y))

        sheet.save(os.path.join(sheet_dir, f"sheet_{i + 1}.jpg"), dpi=(300, 300), quality=95)

        # ✅ Save sheet_1.jpg as preview.png for browser view
        if i == 0:
            preview_path = os.path.join("static", "preview.png")
            sheet.convert("RGB").save(preview_path, dpi=(300, 300), quality=95)

@app.route("/generate-all")
def generate_all():
    generate_cards_and_sheets()
    preview_path = os.path.join("static", "preview.png")
    return send_file(preview_path, mimetype="image/png")


# ✅ Download ZIP for cards
@app.route("/download-cards")
def download_cards():
    from pathlib import Path
    desktop = str(Path.home() / "Desktop")
    card_path = os.path.join(desktop, "cards")
    return create_zip_response(card_path, "cards.zip")

@app.route("/download-sheets")
def download_sheets():
    from pathlib import Path
    desktop = str(Path.home() / "Desktop")
    sheet_path = os.path.join(desktop, "sheets")
    return create_zip_response(sheet_path, "sheets.zip")

# ✅ Helper to zip and return
def create_zip_response(folder_name, zip_filename):
    full_path = os.path.join("static", folder_name)  # 🔁 static se andar ka path

    if not os.path.exists(full_path) or not os.listdir(full_path):
        return f"❌ Folder '{full_path}' is missing or empty", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(full_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, full_path)
                zf.write(file_path, arcname)

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype="application/zip",
        download_name=zip_filename,
        as_attachment=True
    )

        

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)





