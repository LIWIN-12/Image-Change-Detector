# 🖼️ Image Change Detector

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-hf-space-link.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An advanced image comparison tool that detects and highlights differences between two images using computer vision techniques, including SSIM (Structural Similarity Index) and contour detection.

## ✨ Features

- **Dual Image Comparison**: Compare before/after images side by side
- **Smart Change Detection**: Uses SSIM for accurate difference detection
- **Visual Annotations**: Highlights changes with clear visual markers
- **Interactive UI**: User-friendly interface with real-time feedback
- **Responsive Design**: Works on both desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-change-detector.git
   cd image-change-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app_streamlit.py
   ```

## 🖥️ Usage

1. Click "Browse files" to upload your "Before" and "After" images
2. Adjust the sensitivity slider if needed
3. Click "Detect Changes" to analyze the images
4. View the results:
   - **Left Panel**: Original image with marked changes (red rectangles)
   - **Right Panel**: Visual difference map

## 🛠️ Technical Details

### Core Technologies
- **Streamlit** - Web application framework
- **OpenCV** - Image processing and computer vision
- **NumPy** - Numerical computations
- **Pillow** - Image handling
- **scikit-image** - SSIM calculation

### How It Works
1. Images are converted to grayscale and resized to matching dimensions
2. SSIM (Structural Similarity Index) identifies structural differences
3. Contour detection outlines changed regions
4. Results are displayed with visual annotations

## 🌐 Deployment

### Hugging Face Spaces
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

1. Create a new Space on Hugging Face
2. Upload these files:
   - `app_streamlit.py`
   - `requirements.txt`
   - `runtime.txt`
   - `README.md`
3. Configure with Streamlit SDK

### Local Deployment
```bash
docker build -t image-change-detector .
docker run -p 8501:8501 image-change-detector
```

## 📂 Project Structure

```
image-change-detector/
├── app_streamlit.py     # Main Streamlit application
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version specification
├── README.md            # Project documentation
└── tests/               # Test files
    ├── test_app.py
    └── test_core.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Your Name - jkliwinjose@gmail.com

Project Link: (https://huggingface.co/spaces/LIWIN-12/image-change-detector)

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) for powerful computer vision tools
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [scikit-image](https://scikit-image.org/) for SSIM implementation
