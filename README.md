# Project 5: Digital Signal Processing (DSP) Toolkit

![Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.x-blue)

## 📌 Project Overview
Project5DSP is an advanced toolkit developed for processing and analyzing digital signals. The primary goal of this project is to [insert specific goal here, e.g., filter real-time audio or analyze sensor telemetry] using modern DSP algorithms.

This project bridges the gap between mathematical signal theory and practical application by integrating external data sources via APIs, allowing for both local file processing and live data streams.

### Key Features:
* **Spectral Analysis:** Convert signals from the time domain to the frequency domain using FFT (Fast Fourier Transform).
* **Advanced Filtering:** Implementation of custom digital filters (Low-pass, High-pass, Band-pass) to eliminate noise and isolate frequencies.
* **API Integration:** Fetch external datasets or metadata to compare against processed signals.
* **Visual Diagnostics:** Automated generation of spectrograms, Bode plots, and signal comparison charts.

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Primary Libraries:** * `NumPy`: High-performance numerical processing.
    * `SciPy`: Core signal processing toolbox and filter design.
    * `Matplotlib`: Comprehensive data visualization.
    * `python-dotenv`: Management of environment variables and API keys.

---

## ⚙️ Installation & Setup

Follow these steps exactly to set up the project on your local machine:

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/reepchocolade99/Project5DSP.git](https://github.com/reepchocolade99/Project5DSP.git)
cd Project5DSP

2. Install Dependencies
Install the required Python libraries using pip:

```bash
pip install numpy scipy matplotlib python-dotenv

3. API Key & Environment Configuration
This project requires an API key for external data access. For security, this key must be stored in a .env file (which is ignored by Git to prevent leaking your credentials).
In the root directory of the project, create a new file named exactly: .env
Open the .env file in a text editor (e.g., VS Code or Notepad).

Add your credentials in the following format:
---
API_KEY=your_secret_api_key_here
API_ENDPOINT=[https://api.example.com/v1](https://api.example.com/v1)
---
Important: Never share your .env file or commit it to GitHub.

4. Running the Project
Once configured, execute the main script:

```bash
python main.py
