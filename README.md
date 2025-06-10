# 📝 PDF Summarizer (Final Year BE Project)

A full-stack web application that summarizes academic and technical PDFs using a hybrid **BART + Pointer-Generator Network (PGN)** model. The application is built using **Flask (Python)** for the backend and deployed on an **Azure Virtual Machine**.

---

## 👨‍💻 Team Members

1. **Vedant Kokane**  
2. **Ajit Abhyankar**  
3. **Aniket Rajesh**  
4. **Vivek Gotecha**

---

## 🚀 Project Overview

This project helps users extract and summarize content from large academic or research PDFs. It combines the abstractive power of **BART** with the copy-mechanism of **Pointer-Generator Networks (PGN)** for improved handling of domain-specific terminology and long-form input.

### ✨ Key Features

- 🧠 **Hybrid Summarization** using BART + PGN
- 📄 **PDF Upload & Text Extraction** using `spaCyLayout`
- 🗂 **Summary Storage** in SQLite (per user)
- 🔐 **User Authentication** with secure password hashing (`bcrypt`)
- 🕵️ **History View** for tracking past summaries
- ☁️ **Deployed on Azure Virtual Machine**
- ⚙️ **Threaded Background Processing** for PDF summarization

---

## 🧠 Tech Stack

- **Backend**: Python, Flask
- **ML Models**: BART (`facebook/bart-large-cnn`), Pointer-Generator (custom module)
- **PDF Parsing**: spaCy + spaCyLayout
- **Database**: SQLite
- **Authentication**: `bcrypt` password hashing
- **Deployment**: Azure Virtual Machine (Ubuntu)

---
