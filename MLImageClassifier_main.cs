using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using OpenCvSharp;
using Serilog;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

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
        static async Task Main(string[] args)
        {
            // Setup Serilog for logging
            Log.Logger = new LoggerConfiguration()
                .WriteTo.Console()
                .WriteTo.File("logs/log-.txt", rollingInterval: RollingInterval.Day)
                .CreateLogger();

            Log.Information("Application started.");

            if (args.Length == 0)
            {
                Console.WriteLine("Please provide paths to images or directories as command-line arguments.");
                return;
            }

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

            // Process each command-line argument
            foreach (var arg in args)
            {
                if (File.Exists(arg))
                {
                    await ProcessImage(arg, predictionEngine);
                }
                else if (Directory.Exists(arg))
                {
                    var files = Directory.GetFiles(arg, "*.jpg").Concat(Directory.GetFiles(arg, "*.png"));
                    foreach (var file in files)
                    {
                        await ProcessImage(file, predictionEngine);
                    }
                }
                else
                {
                    Log.Error($"Path '{arg}' is neither a file nor a directory.");
                }
            }

            Log.Information("Application ended.");
        }

        // Function to process and predict an image
        private static async Task ProcessImage(string imagePath, PredictionEngine<ImageData, ImagePrediction> predictionEngine)
        {
            try
            {
                Log.Information($"Processing image: {imagePath}");

                // Load and augment the image
                var image = await LoadAndAugmentImage(imagePath);

                // Prepare input data for prediction
                var imageData = new ImageData { ImagePath = imagePath };
                var prediction = predictionEngine.Predict(imageData);

                // Display the image and prediction
                DisplayImageWithPrediction(imagePath, prediction.PredictedLabels);
                LogPrediction(imagePath, prediction.PredictedLabels);

                Log.Information($"Processed image: {imagePath}");
            }
            catch (Exception ex)
            {
                Log.Error($"Error processing image '{imagePath}': {ex.Message}");
            }
        }

        // Function to load and augment image using ImageSharp
        private static async Task<string> LoadAndAugmentImage(string imagePath)
        {
            using (var image = await Image.LoadAsync(imagePath))
            {
                image.Mutate(x => x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                }));

                var augmentedPath = Path.Combine(Path.GetDirectoryName(imagePath), "augmented_" + Path.GetFileName(imagePath));
                await image.SaveAsync(augmentedPath);
                return augmentedPath;
            }
        }

        // Function to display image and prediction using OpenCV
        private static void DisplayImageWithPrediction(string imagePath, float[] predictedLabels)
        {
            var label = predictedLabels.Max().ToString(); // Placeholder for actual label retrieval

            using (var window = new Window("Image Recognition"))
            {
                var image = Cv2.ImRead(imagePath);
                Cv2.PutText(image, $"Prediction: {label}", new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.Red, 2);
                window.Resize(image.Width, image.Height);
                window.ShowImage(image);
                Cv2.WaitKey();
            }
        }

        // Function to log predictions to a file
        private static void LogPrediction(string imagePath, float[] predictedLabels)
        {
            using (StreamWriter writer = new StreamWriter("predictions_log.txt", true))
            {
                writer.WriteLine($"{DateTime.Now}: {imagePath} - {string.Join(",", predictedLabels)}");
            }
        }

        // Function to load pre-trained model and re-train with additional data
        private static ITransformer RetrainModel(MLContext mlContext, string modelPath, IDataView additionalData)
        {
            var loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var trainingPipeline = loadedModel.Transformers.Append(
                mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "Image", imageWidth: 224, imageHeight: 224, inputColumnName: "ImageBytes"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "Pixels", interleavePixelColors: true, offsetImage: 117))
                .Append(mlContext.Model.LoadTensorFlowModel("model.pb").ScoreTensorFlowModel(outputColumnNames: new[] { "Score" }, inputColumnNames: new[] { "Pixels" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
            );
            return trainingPipeline.Fit(additionalData);
        }
    }
}
