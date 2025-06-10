import os
import sqlite3
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, render_template, jsonify,redirect, url_for, flash, session, get_flashed_messages
from werkzeug.utils import secure_filename
import spacy
from spacy_layout import spaCyLayout
import requests
import re
import unicodedata
import threading
from threading import Thread
import bcrypt

from dotenv import load_dotenv

load_dotenv()

# Initialize Flask App
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# secret key
app.secret_key = os.getenv("SECRET_KEY", "fallback-secret")  # Optional fallback


# Model 
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Pointer-Generator Mechanism
class PointerGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pointer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, decoder_hidden, context_vector):
        concat = torch.cat((decoder_hidden, context_vector), dim=-1)
        p_gen = torch.sigmoid(self.pointer(concat))
        return p_gen

# Summarization Model with PGN
class BartWithPointerGenerator(nn.Module):
    def __init__(self, bart_model):
        super().__init__()
        self.bart = bart_model
        self.pointer_generator = PointerGenerator(hidden_dim=1024)  # BART hidden size = 1024

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # Encode input text
        encoder_outputs = self.bart.model.encoder(input_ids, attention_mask=attention_mask)

        # Decode with PGN
        decoder_outputs = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask
        )

        # Get final logits
        logits = self.bart.lm_head(decoder_outputs.last_hidden_state)

        # Compute pointer-generator probability
        p_gen = self.pointer_generator(
            decoder_outputs.last_hidden_state[:, -1, :],  # Last decoder hidden state
            encoder_outputs.last_hidden_state[:, 0, :]  # First encoder hidden state
        )

        return logits, p_gen

# Initialize Model
model = BartWithPointerGenerator(bart_model)


def summarize(text, model, tokenizer, max_length=1024):  # Increase max_length
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)  # Increase input size

    summary_ids = model.bart.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,  # Set max_length higher
        min_length=150,  # Ensure minimum length (optional)
        num_beams=6,  # Increase beams for better quality
        length_penalty=2.0,  # Encourage longer outputs
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Configure Upload Folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# SQLite Database Setup
DATABASE = "database.db"

def init_db():
    """Initializes the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # cursor.execute("DROP TABLE IF EXISTS summaries")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            title TEXT PRIMARY KEY,
            summary TEXT,
            user_email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY NOT NULL,
        password TEXT NOT NULL
    )
""")
    conn.commit()
    conn.close()



# updated

def slugify(title):
     # Step 1: Remove file extension like .pdf, .docx, etc.
    base_title = os.path.splitext(title)[0]

    # Step 2: Lowercase everything
    base_title = base_title.lower()

    # Step 3: Replace all non-alphanumeric characters with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', base_title)

    # Step 4: Remove leading/trailing hyphens
    slug = slug.strip('-')

    return slug

# Load Tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

nlp = spacy.load("en_core_web_sm")
layout = spaCyLayout(nlp)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using spaCyLayout."""
    doc = layout(pdf_path)
    return doc.text.strip()

# Pointer-Generator Mechanism
class PointerGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pointer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, decoder_hidden, context_vector):
        concat = torch.cat((decoder_hidden, context_vector), dim=-1)
        p_gen = torch.sigmoid(self.pointer(concat))
        return p_gen

# BART + Pointer-Generator Model
class BartWithPointerGenerator(nn.Module):
    def __init__(self, bart_model):
        super().__init__()
        self.bart = bart_model
        self.pointer_generator = PointerGenerator(hidden_dim=1024)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_outputs = self.bart.model.encoder(input_ids, attention_mask=attention_mask)
        decoder_outputs = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask
        )
        logits = self.bart.lm_head(decoder_outputs.last_hidden_state)
        p_gen = self.pointer_generator(
            decoder_outputs.last_hidden_state[:, -1, :],  
            encoder_outputs.last_hidden_state[:, 0, :]
        )
        return logits, p_gen

# Load Pre-Trained Model (Saved)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
model = BartWithPointerGenerator(bart_model)
model.eval()

# Summarization Function
def summarize(text, model, tokenizer, max_length=1024):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    summary_ids = model.bart.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=150,
        num_beams=6,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



# Store/Retrieve Summary from Database
def save_summary(title, summary,user_email):
    # updated
    title = slugify(title)
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM summaries WHERE title = ?", (title,))
        result = cursor.fetchone()

        if result:
            print(f"Summary for '{title}' already exists. Skipping insert.")
        else:
            cursor.execute("INSERT INTO summaries (user_email, title, summary) VALUES (?, ?, ?)",(user_email, title, summary))
            conn.commit()
            print("Summary saved successfully.")
    except Exception as e:
        print(f"Error saving summary: {e}")
    finally:
        conn.close()


def process_pdf_and_save(file_path, filename,user_email):
    try:
        # slug = slugify(filename)
        extracted_text = extract_text_from_pdf(file_path)
        if not extracted_text:
            print(f"Extraction failed for {filename}")
            return

        cleaned_text = extracted_text
        summary = summarize(cleaned_text, model, tokenizer)

        save_summary(filename, summary,user_email)
        print(f"Summary stored for: {filename}")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")



# Flask Routes
@app.route("/home")
def index():
    if "user" not in session:
        flash("Please log in to access the home page.")
        return redirect("/")
    return render_template("index.html", email=session["user"])

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/", methods=["POST"])
def handle_login():
    init_db()

    if not request.is_json:
        return jsonify({"ok": False, "error": "Unsupported Content-Type. Expecting JSON"}), 415

    data = request.get_json()
    email = data.get("username")
    password = data.get("password")

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password required"}), 400

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()

    if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
        session["user"] = email
        return jsonify({"ok": True, "redirect": "/home"})
    else:
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    init_db()

    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"ok": False, "error": "Email and password are required."}), 400

        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        with sqlite3.connect(DATABASE, timeout=10) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
            conn.commit()

        flash("Registration successful! You can now log in.", "success")
        return jsonify({"ok": True, "redirect": "/home"})

    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "Email already exists. Please log in."}), 409

    except sqlite3.OperationalError as e:
        return jsonify({"ok": False, "error": f"Database error: {str(e)}"}), 500




# updated code :- 

@app.route("/check_file", methods=["POST"])
def check_file():

    init_db()
    
    data = request.json
    file_name = data.get("fileName")

    if not file_name:
        return jsonify({"error": "File name is required"})
    
    file_name = slugify(file_name)
    # print(f"[DEBUG] Original filename: {file_name}, Slugified: {slug}")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM summaries WHERE title = ?", (file_name,))
    result = cursor.fetchone()
    # print(f"[DEBUG] DB Result: {result}")
    conn.close()
    
    if result:
        response = {"summary": result[0]}
        # print(f"[DEBUG] Returning Response: {response}")
        return jsonify(response)
    else:
        return jsonify({"exists": False, "message": "File not found. Please upload a new file."})

@app.route("/upload", methods=["POST"])
def upload_file():
    init_db()

    if "user" not in session:
        flash("Please log in to upload files.", "error")
        return redirect("/")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    # title = request.form.get('name')
    title = request.form.get('name') 

    if not title:
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(title)
    filename = slugify(filename)

    # Check if already in DB
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM summaries WHERE title = ?", (filename,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return jsonify({"title": filename, "summary": result[0]})  # Already in DB

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Start background thread to process PDF
    user_email = session["user"]
    thread = threading.Thread(target=process_pdf_and_save, args=(file_path, filename,user_email))
    thread.start()

    return jsonify({"status": "processing", "title": filename})  # Respond immediately


@app.route("/logout")
def logout():
    session.clear()  # ✅ Clears all session data
    flash("You have been logged out.", "success")
    return redirect("/")

@app.route("/history")
def history():
    if "user" not in session:
        flash("Please log in to view your history.", "error")
        return redirect(url_for("login"))

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, summary, created_at 
        FROM summaries 
        WHERE user_email = ? 
        ORDER BY created_at DESC
    """, (session["user"],))
    history_data = cursor.fetchall()
    conn.close()

    return render_template("history.html", history=history_data)


@app.route("/all_summaries", methods=["GET"])
def view_all_summaries():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT title, summary FROM summaries")
    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        {"title": row[0], "summary": row[1]} for row in rows
    ])


if __name__ == "__main__":
    app.run(debug=True) 