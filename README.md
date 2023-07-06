# MLImageClassifier
# Image Recognition using ML.NET and OpenCV

This project demonstrates image recognition using machine learning techniques with ML.NET and OpenCV. It utilizes a pre-trained model to classify images and provides real-time predictions.

## Features
- Classify images based on pre-defined labels.
- Load and train the ML.NET model.
- Display images using OpenCV.
- Continuously make predictions on new images.
- User-friendly command-line interface.

## Prerequisites
- .NET Core SDK
- OpenCvSharp NuGet package
- ML.NET NuGet package

## UsageTo use the Image Recognition project:

1. Clone the repository or download the project files.
2. Make sure you have the .NET Core SDK installed on your machine.
3. Install the required NuGet packages:
   - OpenCvSharp: `dotnet add package OpenCvSharp`
   - ML.NET: `dotnet add package Microsoft.ML`
4. Prepare your image dataset and update the `image_dataset.txt` file with the path and corresponding labels for each image.
5. Replace the `model.pb` file with your own pre-trained TensorFlow model or use the provided model.
6. Build the project: `dotnet build`.
7. Run the application: `dotnet run`.

The application will prompt you to enter the path to an image file. It will classify the image and display the predicted label using the OpenCV window. Repeat the process to classify additional images.

Feel free to further customize the code, add error handling, or incorporate additional features based on your requirements.

For more information and detailed instructions, refer to the project's documentation.

Enjoy using the Image Recognition project!
