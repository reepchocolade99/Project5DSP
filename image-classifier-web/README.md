# Amsterdam Parking Violation Analysis System

A Flask-based web application for analyzing parking violation evidence from Dutch municipal towing/parking case report PDFs. The system extracts images from PDFs and uses AI-powered analysis (Claude Vision MLLM or SAM3 segmentation) to generate structured legal reports.

## Features

- **PDF Processing**: Extract embedded images from Dutch parking violation reports (Proces-verbaal)
- **Dual Analysis Modes**:
  - **MLLM (Claude Vision)**: Advanced AI image analysis using Anthropic's Claude API
  - **SAM3**: Segmentation-based object detection (mock mode for development)
- **Bilingual Support**: Full English and Dutch (Nederlands) interface
- **Structured Reports**: Generate legal reports with 7 sections including image description, object detection, location verification, and legal reasoning
- **Evidence Inspector**: Interactive image viewer with detection overlays
- **Export Options**: JSON export, PDF report generation

## Requirements

- Python 3.10+
- pip (Python package manager)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/reepchocolade99/Project5DSP.git
cd Project5DSP/image-classifier-web
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `Flask` - Web framework
- `flask-cors` - Cross-Origin Resource Sharing support
- `python-dotenv` - Environment variable management
- `pymupdf` - PDF processing and image extraction
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `anthropic` - Claude API client (for MLLM mode)

### 4. Configure Environment Variables

Create a `.env` file in the `image-classifier-web` directory:

```bash
touch .env
```

Add the following content to `.env`:

```env
# Anthropic API Key for Claude Vision (MLLM analysis)
ANTHROPIC_API_KEY=your-api-key-here
```

**To get an Anthropic API key:**
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key and copy it

> **Note**: The `.env` file is listed in `.gitignore` and will NOT be committed to the repository.

## Running the Application

### Start the Server

```bash
# Make sure you're in the image-classifier-web directory
cd image-classifier-web

# Run the Flask server
python server.py
```

The server will start at: **http://127.0.0.1:5001**

### Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:5001
```

## Usage

### 1. Upload a PDF

- Click "Choose File" or drag and drop a parking violation PDF
- Supported format: Dutch municipal towing reports (Proces-verbaal wegsleepregeling)

### 2. Configure Settings

- **Language**: Choose English or Nederlands
- **Model Type**:
  - **SAM**: Segmentation-based detection (works offline, mock mode)
  - **MLLM**: Claude Vision AI analysis (requires API key)

### 3. Start Analysis

Click "Start Analysis" to process the document. The system will:
1. Extract embedded images from the PDF
2. Parse structured data (case number, vehicle info, location, violation)
3. Run AI analysis on extracted images
4. Generate a structured legal report

### 4. View Results

The results page has two tabs:
- **Report**: AI-generated analysis with 7 sections
- **Document Summary**: Extracted structured data from the PDF

## Project Structure

```
image-classifier-web/
├── server.py                 # Main Flask application
├── claude_vision_service.py  # Claude Vision API integration
├── sam3_service.py           # SAM3 segmentation service
├── extract_images.py         # PDF image extraction
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this - not in git)
├── .gitignore               # Git ignore rules
│
├── templates/               # Jinja2 HTML templates
│   ├── layout.html          # Base layout
│   ├── index.html           # Upload page
│   └── result.html          # Results page
│
├── static/                  # Static assets
│   ├── styles.css           # Stylesheets
│   ├── app.js               # JavaScript
│   └── gemeente.svg         # Amsterdam logo
│
├── uploads/                 # Uploaded files (temporary)
└── data/                    # Extracted images
    └── derived/             # SAM3 crops and overlays
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page (upload form) |
| `/predict` | POST | Process uploaded PDF |
| `/data/<filename>` | GET | Serve extracted images |
| `/data/derived/<filename>` | GET | Serve SAM3 derived files |
| `/api/export-json` | POST | Export analysis as JSON |

## Troubleshooting

### "MLLM analysis not available"

- Check that `ANTHROPIC_API_KEY` is set in `.env`
- Verify the API key is valid at https://console.anthropic.com/
- Restart the server after adding the key

### "SAM not available - using mock mode"

This is expected behavior. The application runs SAM3 in mock mode by default. For real SAM inference, uncomment the PyTorch dependencies in `requirements.txt` and ensure GPU support.

### PDF extraction fails

- Ensure the PDF is not password-protected
- Check that `pymupdf` is installed correctly
- Try with a different PDF to isolate the issue

### Port already in use

If port 5001 is already in use, either:
- Stop the other process using the port
- Change the port in `server.py` (line: `app.run(debug=True, port=5001)`)

## Development Notes

### Debug Mode

The server runs in debug mode by default, which:
- Auto-reloads on code changes
- Shows detailed error messages

### Adding New Violation Types

Violation codes are defined in `TRANSLATIONS` dict in `server.py`. To add new codes:
1. Add to English translations (`TRANSLATIONS['en']`)
2. Add to Dutch translations (`TRANSLATIONS['nl']`)
3. Add legal template in `dutch_legal_templates` dict

## License

This project is part of a university assignment (DSP Project 5) - Hogeschool van Amsterdam.

---

**Note**: This system is designed for educational and demonstration purposes. Always verify AI-generated legal analysis manually before taking official action.
