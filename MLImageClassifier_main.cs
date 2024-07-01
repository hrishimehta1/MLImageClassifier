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
using System.Collections.Generic;

namespace RetailImageRecognition
{
    public class ProductImageData
    {
        [LoadColumn(0)]
        public string ImagePath { get; set; }

        [LoadColumn(1)]
        public string ProductCategory { get; set; }

        [LoadColumn(2)]
        public string ProductTags { get; set; }
    }

    public class ProductImagePrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels { get; set; }
    }

    public class Startup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers();
            services.AddRazorPages(); // For Blazor WebAssembly
            services.AddServerSideBlazor(); // For Blazor Server
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
                endpoints.MapBlazorHub(); // For Blazor Server
                endpoints.MapFallbackToPage("/_Host"); // For Blazor WebAssembly
            });
        }
    }

    [Route("api/[controller]")]
    [ApiController]
    public class ImageRecognitionController : ControllerBase
    {
        private readonly PredictionEngine<ProductImageData, ProductImagePrediction> _predictionEngine;

        public ImageRecognitionController(PredictionEngine<ProductImageData, ProductImagePrediction> predictionEngine)
        {
            _predictionEngine = predictionEngine;
        }

        [HttpPost]
        public IActionResult Predict([FromBody] ProductImageData imageData)
        {
            var prediction = _predictionEngine.Predict(imageData);
            var categories = new List<string> { "Electronics", "Clothing", "HomeGoods" }; // Example categories
            var predictedCategory = categories[Array.IndexOf(prediction.PredictedLabels, prediction.PredictedLabels.Max())];
            return Ok(new { Category = predictedCategory, Tags = imageData.ProductTags });
        }
    }

    public partial class MainWindow : Window
    {
        private readonly PredictionEngine<ProductImageData, ProductImagePrediction> _predictionEngine;

        public MainWindow(PredictionEngine<ProductImageData, ProductImagePrediction> predictionEngine)
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
                var prediction = await Task.Run(() => _predictionEngine.Predict(new ProductImageData { ImagePath = imagePath }));

                var image = new BitmapImage(new Uri(imagePath));
                SelectedImage.Source = image;
                PredictionLabel.Content = $"Predicted Category: {prediction.PredictedLabels[0]}";
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
            else if (args.Contains("--dashboard"))
            {
                // Run Blazor server for the dashboard
                CreateHostBuilder(args).Build().Run();
            }
            else
            {
                // Run as GUI application
                var mlContext = new MLContext();
                var model = await TrainModel(mlContext);

                var predictionEngine = mlContext.Model.CreatePredictionEngine<ProductImageData, ProductImagePrediction>(model);

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
            var data = mlContext.Data.LoadFromTextFile<ProductImageData>(
                path: "product_image_dataset.txt",
                separatorChar: '\t',
                hasHeader: true);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("ProductCategory")
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
            var containerClient = blobServiceClient.GetBlobContainerClient("product-images");
            var blobClient = containerClient.GetBlobClient(Path.GetFileName(imagePath));

            await blobClient.UploadAsync(imagePath, true);
            return blobClient.Uri.ToString();
        }

        private static async Task ProcessImage(string imagePath, PredictionEngine<ProductImageData, ProductImagePrediction> predictionEngine)
        {
            try
            {
                Log.Information($"Processing image: {imagePath}");

                var augmentedPath = await LoadAndAugmentImage(imagePath);
                var cloudImageUrl = await UploadImageToCloud(augmentedPath);

                var imageData = new ProductImageData { ImagePath = cloudImageUrl };
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

        // Real-time Object Detection using YOLO
        private static void RunObjectDetection()
        {
            var net = CvDnn.ReadNetFromDarknet("yolov3.cfg", "yolov3.weights");
            var layerNames = net.GetLayerNames();
            var outLayerNames = net.GetUnconnectedOutLayersNames();
            using (var cap = new VideoCapture(0))
            {
                if (!cap.IsOpened())
                {
                    throw new Exception("Camera not accessible");
                }

                var frame = new Mat();
                while (true)
                {
                    cap.Read(frame);
                    if (frame.Empty())
                    {
                        break;
                    }

                    var blob = CvDnn.BlobFromImage(frame, 1 / 255.0, new OpenCvSharp.Size(416, 416), new Scalar(0, 0, 0), true, false);
                    net.SetInput(blob);
                    var outputs = new Mat[outLayerNames.Length];
                    net.Forward(outputs, outLayerNames);

                    var classIds = new List<int>();
                    var confidences = new List<float>();
                    var boxes = new List<Rect2d>();

                    for (int i = 0; i < outputs.Length; i++)
                    {
                        var output = outputs[i];
                        for (int j = 0; j < output.Rows; j++)
                        {
                            var scores = output.Row(j).ColRange(5, output.Cols);
                            Cv2.MinMaxLoc(scores, out _, out Point max);
                            var confidence = scores.At<float>(max.Y);

                            if (confidence > 0.5)
                            {
                                var centerX = (int)(output.At<float>(j, 0) * frame.Cols);
                                var centerY = (int)(output.At<float>(j, 1) * frame.Rows);
                                var width = (int)(output.At<float>(j, 2) * frame.Cols);
                                var height = (int)(output.At<float>(j, 3) * frame.Rows);
                                var left = centerX - width / 2;
                                var top = centerY - height / 2;

                                classIds.Add(max.Y);
                                confidences.Add(confidence);
                                boxes.Add(new Rect2d(left, top, width, height));
                            }
                        }
                    }

                    var indices = new List<int>();
                    CvDnn.NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);
                    foreach (var idx in indices)
                    {
                        var box = boxes[idx];
                        Cv2.Rectangle(frame, box, Scalar.Red, 2);
                        var label = $"ID: {classIds[idx]}, Conf: {confidences[idx]:0.00}";
                        Cv2.PutText(frame, label, new Point((int)box.X, (int)box.Y - 10), HersheyFonts.HersheySimplex, 0.5, Scalar.Green, 2);
                    }

                    Cv2.ImShow("Object Detection", frame);
                    if (Cv2.WaitKey(1) == 27) // ESC key
                    {
                        break;
                    }
                }
            }
        }
    }
}
