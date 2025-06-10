# 📝 PDF Summarizer (Final Year BE Project)

---

## 🌟 Project Overview
PDF Summarizer is a web application that extracts text from PDF files and generates concise, high-quality summaries using the **BART model** (facebook/bart-large-cnn) enhanced with a custom **Pointer-Generator Network (PGN)**. Built with **Flask** and **SQLite**, it offers user authentication, file upload, and summary history features. The project is deployed on an **Azure Virtual Machine** for reliable access.

This project was crafted as a final-year project by:

1. **Vedant Kokane**  
2. **Ajit Abhyankar**  
3. **Aniket Rajesh**  
4. **Vivek Gotecha**

---

## ✨ Features
- 📑 **PDF Text Extraction**: Extracts text from PDFs using spaCy with a custom `spaCyLayout` module.  
- ✍️ **Advanced Summarization**: Combines BART with a Pointer-Generator Network for precise, context-aware summaries.  
- 🔒 **User Authentication**: Secure login/register system with bcrypt password hashing.  
- 💾 **Database Storage**: Stores summaries and user data in SQLite with unique slugs for filenames.  
- ⬆️ **File Upload**: Supports PDF uploads (up to 16MB) with asynchronous processing.  
- 📜 **History Tracking**: Displays a history of summarized PDFs for each user.  
- ☁️ **Azure VM Deployment**: Hosted on an Azure Virtual Machine for scalability and accessibility.

---

## 🛠️ Tech Stack
- **Backend**: Flask (Python)  
- **Machine Learning**: PyTorch, Transformers (BART), spaCy  
- **Database**: SQLite  
- **Frontend**: HTML, CSS, JavaScript (Flask templates)  
- **Authentication**: bcrypt  
- **Deployment**: Azure Virtual Machine  
- **Other Libraries**: werkzeug, python-dotenv, requests

---

## 🚀 Usage
1. **Register/Login**: Create an account or log in to access the summarizer.  
2. **Upload PDF**: Upload a PDF file with a title on the homepage.  
3. **View Summary**: Summaries are generated in the background and stored in the database.  
4. **Check History**: View all your summarized PDFs in the History section.  
5. **Logout**: End your session securely.

---

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

## 🔍 Notes
- **File Size Limit**: Supports PDFs up to 16MB.  
- **Asynchronous Processing**: Summarization runs in the background for a smooth user experience.  
- **Security**: Passwords are hashed with bcrypt, and filenames are sanitized for safety.  
- **Azure Hosting**: Deployed on an Azure VM for reliable access.

---

## 🌱 Future Improvements
- Support for additional file formats (e.g., DOCX, TXT).  
- Enhanced text preprocessing for better summarization.  
- Modern frontend with React or similar frameworks.  
- API endpoints for programmatic access.

---

## 🙌 Acknowledgments
- **Transformers**: For the BART model.  
- **spaCy**: For robust PDF text extraction.  
- **Flask**: For a lightweight web framework.  
- **Azure**: For seamless cloud hosting.

---

## 👥 Team
Developed with passion by:  
- **Vedant Kokane**  
- **Ajit Abhyankar**  
- **Aniket Rajesh**  
- **Vivek Gotecha**

---
