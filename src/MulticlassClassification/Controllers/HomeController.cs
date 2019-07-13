using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using HandwritingRecognitionML;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using MulticlassClassificationML.Model.DataModels;

namespace MulticlassClassification.Controllers
{
    public class HomeController : Controller
    {
        private const int SizeOfImage = 32;
        private const int SizeOfArea = 4;
        private const string ModelName = "MLModel.zip";
        HandwritingRecognition handwritingRecognition;

        public HomeController()
        {
            handwritingRecognition = new HandwritingRecognition();
        }

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Upload(string imgBase64)
        {
            if (string.IsNullOrEmpty(imgBase64))
            {
                return BadRequest(new { prediction = "-", dataset = string.Empty });
            }
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(ModelName, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            var datasetValue = GetDatasetValuesFromImage(imgBase64);

            var input = new ModelInput();
            input.PixelValues = datasetValue.ToArray();
            ModelOutput result = predEngine.Predict(input);

            return Ok(new { prediction = (char)(result.Prediction + 'A'), dataset = string.Join(",", datasetValue) });
        }

        private List<float> GetDatasetValuesFromImage(string base64Image)
        {
            base64Image = base64Image.Replace("data:image/png;base64,", "");
            var imageBytes = Convert.FromBase64String(base64Image).ToArray();

            Image image;

            using (var stream = new MemoryStream(imageBytes))
            {
                image = Image.FromStream(stream);
            }

            var res = new Bitmap(SizeOfImage, SizeOfImage);
            using (var g = Graphics.FromImage(res))
            {
                g.Clear(Color.White);
                g.DrawImage(image, 0, 0, SizeOfImage, SizeOfImage);
            }

            var datasetValue = handwritingRecognition.GetDatasetValues(res, "ffffffff");

            return datasetValue;
        }
    }
}
