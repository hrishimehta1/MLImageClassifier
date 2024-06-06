using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using OpenCvSharp;
using Serilog;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using Azure.Storage.Blobs;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace ImageRecognition
{
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels { get; set; }
    }

    public class Startup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers();
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            app.UseRouting();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }

    [Route("api/[controller]")]
    [ApiController]
    public class ImageRecognitionController : ControllerBase
    {
        private readonly PredictionEngine<ImageData, ImagePrediction> _predictionEngine;

        public ImageRecognitionController(PredictionEngine<ImageData, ImagePrediction> predictionEngine)
        {
            _predictionEngine = predictionEngine;
        }

        [HttpPost]
        public IActionResult Predict([FromBody] ImageData imageData)
        {
            var prediction = _predictionEngine.Predict(imageData);
            return Ok(prediction.PredictedLabels);
        }
    }

    public partial class MainWindow : Window
    {
        private readonly PredictionEngine<ImageData, ImagePrediction> _predictionEngine;

        public MainWindow(PredictionEngine<ImageData, ImagePrediction> predictionEngine)
        {
            InitializeComponent();
            _predictionEngine = predictionEngine;
        }

        private async void OnSelectImage(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Image files (*.png;*.jpg)|*.png;*.jpg"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                var imagePath = openFileDialog.FileName;
                var prediction = await Task.Run(() => _predictionEngine.Predict(new ImageData { ImagePath = imagePath }));

                var image = new BitmapImage(new Uri(imagePath));
                SelectedImage.Source = image;
                PredictionLabel.Content = $"Predicted Label: {prediction.PredictedLabels[0]}";
            }
        }
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

            if (args.Contains("--server"))
            {
                // Run as web API server
                CreateHostBuilder(args).Build().Run();
            }
            else
            {
                // Run as GUI application
                var mlContext = new MLContext();
                var model = await TrainModel(mlContext);

                var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

                // Start WPF GUI
                var app = new Application();
                var mainWindow = new MainWindow(predictionEngine);
                app.Run(mainWindow);
            }

            Log.Information("Application ended.");
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });

        private static async Task<ITransformer> TrainModel(MLContext mlContext)
        {
            var data = mlContext.Data.LoadFromTextFile<ImageData>(
                path: "image_dataset.txt",
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: "ImageBytes", imageFolder: null, inputColumnName: "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "Image", imageWidth: 224, imageHeight: 224, inputColumnName: "ImageBytes"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "Pixels", interleavePixelColors: true, offsetImage: 117))
                .Append(mlContext.Model.LoadTensorFlowModel("model.pb").ScoreTensorFlowModel(outputColumnNames: new[] { "Score" }, inputColumnNames: new[] { "Pixels" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);

            return model;
        }

        private static async Task<string> UploadImageToCloud(string imagePath)
        {
            var blobServiceClient = new BlobServiceClient("your_connection_string_here");
            var containerClient = blobServiceClient.GetBlobContainerClient("images");
            var blobClient = containerClient.GetBlobClient(Path.GetFileName(imagePath));

            await blobClient.UploadAsync(imagePath, true);
            return blobClient.Uri.ToString();
        }

        private static async Task ProcessImage(string imagePath, PredictionEngine<ImageData, ImagePrediction> predictionEngine)
        {
            try
            {
                Log.Information($"Processing image: {imagePath}");

                var augmentedPath = await LoadAndAugmentImage(imagePath);
                var cloudImageUrl = await UploadImageToCloud(augmentedPath);

                var imageData = new ImageData { ImagePath = cloudImageUrl };
                var prediction = predictionEngine.Predict(imageData);

                DisplayImageWithPrediction(imagePath, prediction.PredictedLabels);
                LogPrediction(imagePath, prediction.PredictedLabels);

                Log.Information($"Processed image: {imagePath}");
            }
            catch (Exception ex)
            {
                Log.Error($"Error processing image '{imagePath}': {ex.Message}");
            }
        }

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

        private static void DisplayImageWithPrediction(string imagePath, float[] predictedLabels)
        {
            var label = predictedLabels.Max().ToString();

            using (var window = new Window("Image Recognition"))
            {
                var image = Cv2.ImRead(imagePath);
                Cv2.PutText(image, $"Prediction: {label}", new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.Red, 2);
                window.Resize(image.Width, image.Height);
                window.ShowImage(image);
                Cv2.WaitKey();
            }
        }

        private static void LogPrediction(string imagePath, float[] predictedLabels)
        {
            using (StreamWriter writer = new StreamWriter("predictions_log.txt", true))
            {
                writer.WriteLine($"{DateTime.Now}: {imagePath} - {string.Join(",", predictedLabels)}");
            }
        }
    }
}
