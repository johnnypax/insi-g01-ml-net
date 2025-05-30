//L'obiettivo è prevedere il ritardo (in minuti - Delay) di un autobus in base a : Orario di partenza (DeapartureHour), Meteo (Weather), Si tratta di week end (IsWeekEnd)


using Microsoft.ML;
using Microsoft.ML.Data;

var dataSet = new List<BusData>
{
    new BusData() { DepartureHour = 8, Weather = "Sunny", IsWeekend = false, DelayMinutes = 2 },
    new BusData() { DepartureHour = 9, Weather = "Rainy", IsWeekend = false, DelayMinutes = 5 },
    new BusData() { DepartureHour = 7, Weather = "Unknown", IsWeekend = false, DelayMinutes = 9 },
    new BusData() { DepartureHour = 17, Weather = null, IsWeekend = true, DelayMinutes = 2 },
    new BusData() { DepartureHour = 19, Weather = "Foggy", IsWeekend = true, DelayMinutes = 5 },
};

var mlContext = new MLContext();

var traningData = mlContext.Data.LoadFromEnumerable(dataSet);

// Features = [DepartureNormalized, WeekendEncoded, WeatherEncoded]

var pipeline = mlContext.Transforms
    .NormalizeMinMax("DepartureNormalized", "DepartureHour")
    .Append(mlContext.Transforms.Conversion.ConvertType("WeekendEncoded", "IsWeekend", DataKind.Single))
    .Append(mlContext.Transforms.CustomMapping<BusData, BusDataCleaned>((input, output) =>
    {
        output.Weather = string.IsNullOrEmpty(input.Weather) || input.Weather == "Unknown" ? "Other" : input.Weather;

        output.DepartureHour = input.DepartureHour;
        output.IsWeekend = input.IsWeekend;
        output.DelayMinutes = input.DelayMinutes;
    }, contractName: "custom_filter"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("WeatherEncoded", "Weather")
    .Append(mlContext.Transforms.Concatenate("Features",
                "DepartureNormalized", "WeekendEncoded", "WeatherEncoded"))
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "DelayMinutes", maximumNumberOfIterations: 500)));

var model = pipeline.Fit(traningData);

var predEngine = mlContext.Model.CreatePredictionEngine<BusData, DelayPrediction>(model);

var prediction = predEngine.Predict(new BusData()
{
    DepartureHour = 9,
    Weather = "Foggy",
    IsWeekend = true,
});

Console.WriteLine($"Ritardo previsto: {prediction.Score:F2} m");


#region Classes
public class BusData
{
    public float DepartureHour { get; set; }        //Es. 7, 14, 18
    public string? Weather { get; set; }            //"Sunny", "Rainy", "Foggy", ecc..
    public bool IsWeekend { get; set; }             //true/false
    public float DelayMinutes { get; set; }         //target - Variabile DIP
}

public class BusDataCleaned : BusData { }

public class DelayPrediction
{
    public float Score { get; set; }
}


#endregion