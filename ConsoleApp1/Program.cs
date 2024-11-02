using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis
{
    public class NewsData
    {
        [LoadColumn(0)]
        public string Headline { get; set; }

        [LoadColumn(1)]
        public bool Label { get; set; }
    }

    public class NewsPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    public class NewsApiResponse
    {
        public List<Article> Articles { get; set; }
    }

    public class Article
    {
        public string Title { get; set; }
        public string Description { get; set; }
    }

    class Program
    {
        private static readonly string apiKey = "1039938563e34e99a036618203764243";
        private static readonly string apiUrl = $"https://newsapi.org/v2/top-headlines?category=general&apiKey={apiKey}";
        private static string filePath = @"/financial_news.csv";

        private static MLContext mlContext = new MLContext();
        private static ITransformer model;
        private static PredictionEngine<NewsData, NewsPrediction> predictor;

        static async Task Main(string[] args)
        {
            LoadAndTrainModel();

            bool hasExited = false;
            while (!hasExited)
            {
                Console.WriteLine("Choose an option: \n 1- Enter a news headline \n 2- Fetch latest headlines from News API \n 3- Exit");
                var option = Console.ReadLine();
                var existingHeadlines = LoadExistingHeadlines();


                switch (option)
                {
                    case "1":
                        Console.WriteLine("Enter a news headline:");
                        var headline = Console.ReadLine();
                        var prediction = predictor.Predict(new NewsData { Headline = headline });

                        Console.WriteLine($"Prediction: {(prediction.Prediction ? "Positive" : "Negative")}, Probability: {prediction.Probability}");

                        Console.WriteLine("Is this prediction correct? (yes/no): ");
                        var feedback = Console.ReadLine().ToLower();

                        if (feedback == "no")
                        {
                            Console.WriteLine("Please provide the correct label (1 for Positive, 0 for Negative): ");
                            bool correctLabel = Console.ReadLine() == "1";
                            SaveNewData(headline, correctLabel);
                            RetrainModel();
                        }
                        else if (feedback == "yes" && !existingHeadlines.Contains(headline))
                        {
                            Console.WriteLine("Feedback received. Thank you!");
                            bool correctLabel = prediction.Prediction;
                            SaveNewData(headline, correctLabel);
                            RetrainModel();
                        }
                        break;

                    case "2":
                        var headlines = await FetchHeadlines();

                        foreach (var fetchedHeadline in headlines)
                        {
                            if (!existingHeadlines.Contains(fetchedHeadline))
                            {
                                Console.WriteLine($"Fetched Headline: {fetchedHeadline}");
                                Console.WriteLine("Please provide the correct label (1 for Positive, 0 for Negative): ");
                                bool label = Console.ReadLine() == "1";
                                SaveNewData(fetchedHeadline, label);
                                existingHeadlines.Add(fetchedHeadline);
                            }
                            else
                            {
                                Console.WriteLine("Headline already exists.");
                            }
                        }
                        RetrainModel();
                        break;

                    case "3":
                        hasExited = true;
                        break;

                    default:
                        Console.WriteLine("Invalid option. Please try again.");
                        break;
                }
            }
        }

        static HashSet<string> LoadExistingHeadlines()
        {
            var existingHeadlines = new HashSet<string>();

            if (File.Exists(filePath))
            {
                using var reader = new StreamReader(filePath);
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    if (line != null)
                    {
                        var commaIndex = line.LastIndexOf(',');
                        if (commaIndex > 0)
                        {
                            var headline = line.Substring(0, commaIndex).Trim('"');
                            existingHeadlines.Add(headline);
                        }
                    }
                }
            }

            return existingHeadlines;
        }


        static void LoadAndTrainModel()
        {
            IDataView trainingData = mlContext.Data.LoadFromTextFile<NewsData>(
                path: filePath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            var options = new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = "Label",
                FeatureColumnName = "Features",
                Shuffle = false
            };

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(NewsData.Headline))
                            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(options));

            model = pipeline.Fit(trainingData);
            predictor = mlContext.Model.CreatePredictionEngine<NewsData, NewsPrediction>(model);
            Console.WriteLine("Model trained successfully.");
        }

        private static async Task<List<string>> FetchHeadlines()
        {
            List<string> headlines = new List<string>();

            using HttpClient client = new HttpClient();
            client.DefaultRequestHeaders.Add("User-Agent", "C# App");
            client.DefaultRequestHeaders.Add("Accept", "application/json");

            try
            {
                HttpResponseMessage response = await client.GetAsync(apiUrl);

                if (!response.IsSuccessStatusCode)
                {
                    // Log detailed response message if not successful
                    string errorContent = await response.Content.ReadAsStringAsync();
                    throw new HttpRequestException($"Request failed: {response.StatusCode}\nContent: {errorContent}");
                }

                var apiResponse = await response.Content.ReadFromJsonAsync<NewsApiResponse>();

                if (apiResponse?.Articles != null)
                {
                    foreach (var article in apiResponse.Articles)
                    {
                        if (!string.IsNullOrEmpty(article.Title))
                        {
                            headlines.Add(article.Title);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred while fetching headlines: " + ex.Message);
                throw;
            }

            return headlines;
        }

        static void SaveNewData(string headline, bool label)
        {
            using (StreamWriter sw = File.AppendText(filePath))
            {
                sw.WriteLine($"\"{headline}\",{(label ? 1 : 0)}");
            }
            Console.WriteLine("New data added to training set.");
        }

        static void RetrainModel()
        {
            Console.WriteLine("Retraining model with new data...");
            LoadAndTrainModel();
        }
    }
}
