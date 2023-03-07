using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using Common;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    internal class Program
    {
        public static string cifar10Url = @"https://github.com/YoongiKim/CIFAR-10-images/archive/refs/heads/master.zip";

        static void Main()
        {
            var seed = 0;
            var cifar10FolderPath = Path.Join(Path.GetTempPath(), "cifar");
            var cifar10zipPath = Path.Join(cifar10FolderPath, "cifar10.zip");
            if (!Directory.Exists(cifar10FolderPath))
            {
                Directory.CreateDirectory(cifar10FolderPath);
                // download
                var client = new WebClient();
                client.DownloadFile(cifar10Url, cifar10zipPath);
                ZipFile.ExtractToDirectory(cifar10zipPath, cifar10FolderPath);
            }
            var imageInputs = Directory.GetFiles(cifar10FolderPath)
                .Where(p => Path.GetExtension(p) == ".jpg")
                .Select(p => new ModelInput
                {
                    ImagePath = p,
                    Label = p.Split("\\").SkipLast(1).Last(),
                });

            var testImages = imageInputs.Where(f => f.ImagePath.Contains("test"));
            var trainImages = imageInputs.Where(f => f.ImagePath.Contains("train"));
            var context = new MLContext(seed);
            context.Log += FilterMLContextLog;
            var trainDataset = context.Data.LoadFromEnumerable(trainImages);
            var testDataset = context.Data.LoadFromEnumerable(testImages);

            var pipeline = context.Transforms.LoadRawImageBytes("Features", null, "ImagePath")
                            .Append(context.Transforms.Conversion.MapValueToKey("Label"))
                            .Append(context.MulticlassClassification.Trainers.ImageClassification())
                            .Append(context.Transforms.Conversion.MapValueToKey("Label"));


            var model = pipeline.Fit(trainDataset);
            var eval = model.Transform(testDataset);
            var metric = context.MulticlassClassification.Evaluate(eval);

            Console.Write($"Accuracy: {metric.MacroAccuracy}");
        }

        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }

        class ModelInput
        {
            public string ImagePath { get; set; }

            public string Label { get; set; }
        }
    }
}

