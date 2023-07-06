using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace ImageRecognition
{
    // Represents the input data schema for image classification
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    // Represents the output prediction schema
    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Set up MLContext
            var mlContext = new MLContext();

            // Load and train the ML.NET model
            var data = mlContext.Data.LoadFromTextFile<ImageData>(
                path: "image_dataset.txt",
                separatorChar: '\t',
                hasHeader: true);

            // Define the ML.NET pipeline for image classification
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "Image", imageWidth: 224, imageHeight: 224, inputColumnName: "ImageBytes"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "Pixels", interleavePixelColors: true, offsetImage: 117))
                .Append(mlContext.Model.LoadTensorFlowModel("model.pb").ScoreTensorFlowModel(outputColumnNames: new[] { "Score" }, inputColumnNames: new[] { "Pixels" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);

            // Create a prediction engine to make predictions on new images
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            while (true)
            {
                Console.WriteLine("Enter the path to an image file (or 'exit' to quit):");
                var input = Console.ReadLine();

                if (input == "exit")
                    break;

                if (!File.Exists(input))
                {
                    Console.WriteLine("File not found. Please try again.");
                    continue;
                }

                // Prepare input data for prediction
                var imageData = new ImageData { ImagePath = input };
                var prediction = predictionEngine.Predict(imageData);
                var predictedLabel = prediction.PredictedLabels[0];

                Console.WriteLine($"Predicted label for {input}: {predictedLabel}");

                // Display the image using OpenCV
                using (var window = new Window("Image Recognition"))
                {
                    var image = Cv2.ImRead(input);
                    window.Resize(image.Width, image.Height);
                    window.ShowImage(image);
                    Cv2.WaitKey();
                }
            }
        }
    }
}
