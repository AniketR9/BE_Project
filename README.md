# 📝 PDF Summarizer (Final Year BE Project)

---

## 🌟 Project Overview

**PDF Summarizer** is a web application designed to extract text from PDF files and generate concise, high-quality summaries using the **BART model** (`facebook/bart-large-cnn`) enhanced with a custom **Pointer-Generator Network (PGN)**. Built with **Flask** and **SQLite**, it includes features like user authentication, file upload, and a personalized summary history. The project is deployed on an **Azure Virtual Machine** for scalable and reliable access.

> 🎓 Developed as a final-year engineering project by:
1. **Vedant Kokane**  
2. **Ajit Abhyankar**  
3. **Aniket Rajesh**  
4. **Vivek Gotecha**

---

## ✨ Key Features

- 📑 **PDF Text Extraction**: Utilizes `spaCy` with a custom `spaCyLayout` module to extract structured text from PDFs.
- 🧠 **Advanced Summarization**: Combines BART with a custom PGN to produce accurate, context-aware summaries.
- 🔐 **User Authentication**: Secure login and registration with password hashing using `bcrypt`.
- 💾 **Database Integration**: Stores user data and summaries using `SQLite`, with unique slugs for efficient retrieval.
- ⬆️ **File Upload**: Accepts PDF files up to 16MB with background processing for smooth UX.
- 🕓 **History Tracking**: Lets users view and revisit their previously summarized files.
- ☁️ **Cloud Deployment**: Hosted on an Azure Virtual Machine for global access and uptime.

---

## 🛠️ Tech Stack

| Layer            | Tools & Libraries                          |
|------------------|--------------------------------------------|
| **Backend**       | Flask (Python)                             |
| **ML Models**     | HuggingFace Transformers (BART), PyTorch, PGN |
| **NLP**           | spaCy + spaCyLayout                        |
| **Database**      | SQLite                                     |
| **Authentication**| bcrypt                                     |
| **Frontend**      | HTML, CSS, JavaScript (Flask templates)    |
| **Deployment**    | Azure Virtual Machine                      |
| **Other Tools**   | `werkzeug`, `python-dotenv`, `requests`    |

---

## 🚀 How to Use

1. 🔐 **Register/Login**: Create an account to start using the app.
2. 📤 **Upload PDF**: Provide a title and upload your PDF document.
3. 🧠 **Get Summary**: The summarization process runs in the background.
4. 📜 **View History**: Access previously generated summaries in the "History" tab.
5. 🔓 **Logout**: End your session securely at any time.

---

## 📁 Project Structure


## 📂 Project Structure
```
pdf-summarizer/
├── app.py                # Main Flask application
├── uploads/              # Folder for uploaded PDFs
├── templates/            # HTML templates (index.html, login.html, register.html, history.html, about.html)
├── database.db           # SQLite database (auto-created)
├── .env                 # Environment variables
└── README.md            # This file
```

---

## 🔐 Notes & Best Practices

- 📄 **Max File Size**: PDF uploads are capped at 16MB.
- ⚙️ **Asynchronous Tasks**: Summarization is performed in a background thread for responsiveness.
- 🛡️ **Security First**: Passwords are hashed, and uploaded filenames are sanitized to prevent security issues.
- 🌐 **Azure Hosting**: Application is deployed on a reliable Azure VM.

---

## 🌱 Future Enhancements

- ✅ Support for additional formats: DOCX, TXT, Markdown
- ✅ Pre-summarization cleanup for noisy PDFs
- ✅ REST API for external integration
- ✅ Improved frontend using React, Vue, or Next.js
- ✅ Switch to PostgreSQL for production-grade storage

---

## 🙌 Acknowledgments

Special thanks to the amazing open-source tools and platforms that made this project possible:

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [spaCy NLP](https://spacy.io/)
- [Flask Web Framework](https://flask.palletsprojects.com/)
- [Microsoft Azure](https://azure.microsoft.com/)

---
