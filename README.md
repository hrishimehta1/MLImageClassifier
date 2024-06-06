# MLImageClassifier

**Image Recognition using ML.NET, OpenCV, WPF, ASP.NET Core, and Azure Blob Storage**

This project demonstrates advanced image recognition using machine learning techniques with ML.NET and OpenCV. It utilizes a pre-trained model to classify images and provides real-time predictions. The project also features a graphical user interface, web API integration, cloud storage, and more.

## Features

- Classify images based on pre-defined labels.
- Load and train the ML.NET model.
- Display images using OpenCV.
- Continuously make predictions on new images.
- User-friendly graphical user interface (GUI) using WPF.
- RESTful API for making predictions via HTTP requests using ASP.NET Core.
- Cloud storage integration with Azure Blob Storage.
- Advanced image processing and augmentation.
- Multi-threading for concurrent image processing.
- Model management with versioning and retraining capabilities.
- Dockerized deployment for easy deployment and scalability.

## Prerequisites

- .NET Core SDK
- OpenCvSharp NuGet package
- ML.NET NuGet package
- Azure Storage Blob NuGet package
- Serilog NuGet package
- SixLabors.ImageSharp NuGet package
- Docker (optional, for containerized deployment)

## Usage

To use the Image Recognition project:

1. Clone the repository or download the project files.
2. Make sure you have the .NET Core SDK installed on your machine.
3. Install the required NuGet packages:
    - OpenCvSharp: `dotnet add package OpenCvSharp`
    - ML.NET: `dotnet add package Microsoft.ML`
    - Azure Storage Blob: `dotnet add package Azure.Storage.Blobs`
    - Serilog: `dotnet add package Serilog`
    - SixLabors.ImageSharp: `dotnet add package SixLabors.ImageSharp`
4. Prepare your image dataset and update the `image_dataset.txt` file with the path and corresponding labels for each image.
5. Replace the `model.pb` file with your own pre-trained TensorFlow model or use the provided model.
6. Build the project: `dotnet build`.

### Running the Application

#### Graphical User Interface (GUI)

To run the application with a GUI:

1. Run the application: `dotnet run`
2. Use the GUI to select and classify images. The application will display the image and the predicted label using OpenCV.

#### Command-Line Interface (CLI)

To run the application with a command-line interface:

1. Run the application: `dotnet run -- [image-path]`
2. Enter the path to an image file when prompted. The application will classify the image and display the predicted label using OpenCV.

#### Web API

To run the application as a web API:

1. Run the application with the server flag: `dotnet run --server`
2. Use a tool like Postman to send POST requests with image data to the API endpoint: `http://localhost:5000/api/ImageRecognition`

### Docker Deployment

To deploy the application using Docker:

1. Build the Docker image: `docker build -t mlimageclassifier .`
2. Run the Docker container: `docker run -p 5000:80 mlimageclassifier`

The application will be accessible via `http://localhost:5000`.

## Advanced Features

### Cloud Storage

The application supports cloud storage integration with Azure Blob Storage. Images are uploaded to Azure Blob Storage before classification.

### Logging and Error Handling

The application uses Serilog for comprehensive logging and error handling. Logs are written to both console and files.

### Multi-threading

The application uses multi-threading to process multiple images concurrently, improving performance and efficiency.

### Model Management

The application includes model management features, allowing for model versioning and retraining with additional data.

### Image Processing and Augmentation

The application uses SixLabors.ImageSharp for advanced image processing and augmentation, ensuring high-quality inputs for the model.

Feel free to further customize the code, add error handling, or incorporate additional features based on your requirements.

For more information and detailed instructions, refer to the project's documentation.

Enjoy using the Image Recognition project!
