# MLImageClassifier

**Image Recognition using ML.NET, OpenCV, WPF, ASP.NET Core, Azure Blob Storage, and Blazor**

This project demonstrates advanced image recognition using machine learning techniques with ML.NET and OpenCV. It utilizes a pre-trained model to classify images and provides real-time predictions. The project also features a graphical user interface, web API integration, cloud storage, and more.

## Features

- **Classify Images**: Automatically categorize products and assign relevant tags.
- **ML Model Training**: Train and fine-tune the ML.NET model with a custom product image dataset.
- **Real-Time Predictions**: Continuously make predictions on new images using the web API and GUI.
- **Advanced Image Processing**: Utilize OpenCV and SixLabors.ImageSharp for image manipulation and augmentation.
- **User-Friendly GUI**: WPF-based graphical interface for image selection and display.
- **RESTful API**: ASP.NET Core web API for making predictions via HTTP requests.
- **Cloud Integration**: Upload and store images in Azure Blob Storage for scalable storage solutions.
- **Logging and Error Handling**: Comprehensive logging with Serilog for tracking application performance and errors.
- **Multi-Threading**: Concurrent image processing for improved performance.
- **Model Management**: Support for model versioning and retraining with new data.
- **Docker Deployment**: Containerize the application for easy deployment and scalability.
- **Real-Time Object Detection**: Integrated real-time object detection using YOLO with OpenCvSharp.
- **Batch Processing**: Implemented batch image processing for large-scale operations.
- **Web-Based Dashboard**: Blazor-based dashboard for monitoring and managing the image processing pipeline.
- **Custom Model Training**: Interface for users to upload their own datasets and train custom models.
- **Automated Model Evaluation**: Evaluate model performance with metrics like accuracy, precision, recall, and F1 score.

## Prerequisites

- .NET Core SDK
- OpenCvSharp NuGet package
- ML.NET NuGet package
- Azure Storage Blob NuGet package
- Serilog NuGet package
- SixLabors.ImageSharp NuGet package
- Docker (optional, for containerized deployment)

## Usage

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/MLImageClassifier.git
   cd MLImageClassifier
# Installation and Setup

## Install NuGet Packages

```bash
dotnet add package OpenCvSharp
dotnet add package Microsoft.ML
dotnet add package Azure.Storage.Blobs
dotnet add package Serilog
dotnet add package SixLabors.ImageSharp
dotnet add package Microsoft.AspNetCore.Components.WebAssembly.Server
dotnet add package Microsoft.AspNetCore.Components.WebAssembly
```

## Prepare Dataset

- Update `product_image_dataset.txt` with the path and corresponding labels for each image.
- Replace `model.pb` with your own pre-trained TensorFlow model or use the provided model.

## Build the Project

```bash
dotnet build
```

# Running the Application

## Graphical User Interface (GUI)

To run the application with a GUI:

```bash
dotnet run
```
Use the GUI: Select and classify images. The application will display the image and the predicted category using OpenCV.

## Command-Line Interface (CLI)

To run the application with a command-line interface:

```bash
dotnet run -- [image-path]
```
Enter Image Path: Enter the path to an image file when prompted. The application will classify the image and display the predicted category using OpenCV.

## Web API

To run the application as a web API:

```bash
dotnet run --server
```
Use Postman or Curl: Send POST requests with image data to the API endpoint:

```bash
curl -X POST "http://localhost:5000/api/ImageRecognition" -H "Content-Type: application/json" -d '{"ImagePath":"path/to/image.jpg"}'
```

## Web-Based Dashboard

To run the Blazor-based web dashboard:

```bash
dotnet run --dashboard
```
Access the Dashboard: Open your web browser and go to `http://localhost:5000` to use the Blazor-based dashboard for monitoring and managing the image processing pipeline.

# Docker Deployment

## Build Docker Image

```bash
docker build -t mlimageclassifier .
```

## Run Docker Container

```bash
docker run -p 5000:80 mlimageclassifier
```
The application will be accessible via `http://localhost:5000`.

# Advanced Features

## Real-Time Object Detection

The application includes real-time object detection using YOLO and OpenCvSharp. This feature allows for the detection and labeling of objects in real-time using a webcam feed.

## Batch Processing

Batch image processing is implemented to handle large-scale operations. This allows for the concurrent processing of multiple images, improving efficiency and scalability.

## Custom Model Training

Users can upload their own datasets and train custom models through the application. This feature allows for the flexibility to adapt the model to specific requirements and data.

## Automated Model Evaluation

The application includes automated model evaluation capabilities, providing metrics such as accuracy, precision, recall, and F1 score. This ensures that the model's performance can be tracked and improved over time.

For more information and detailed instructions, refer to the project's documentation.

Enjoy using the Image Recognition project!
