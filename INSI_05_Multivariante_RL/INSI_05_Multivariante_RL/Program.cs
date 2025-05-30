/* 
 * Creare una correlazione tra:
 * - Ore di sole
 * - Temperatura dell'aria
 * - Inclinazione del pannello solare
 * - Guadagno in euro
 */

using Microsoft.ML;

var dataSet = new List<DatiPannello>
{
    new DatiPannello(){ OreDiSole = 5, Temperatura = 25, Inclinazione = 30, Guadagno = 150 },
    new DatiPannello(){ OreDiSole = 6, Temperatura = 28, Inclinazione = 35, Guadagno = 180 },
    new DatiPannello(){ OreDiSole = 7, Temperatura = 30, Inclinazione = 30, Guadagno = 210 },
    new DatiPannello(){ OreDiSole = 8, Temperatura = 32, Inclinazione = 25, Guadagno = 240 },
    new DatiPannello(){ OreDiSole = 9, Temperatura = 33, Inclinazione = 22, Guadagno = 250 },
    new DatiPannello(){ OreDiSole = 10, Temperatura = 35, Inclinazione = 30, Guadagno = 200 },
};

var mlContext = new MLContext();

var trainingData = mlContext.Data.LoadFromEnumerable(dataSet);

//Features = [OreDiSole, Temperatura, Inclinazione]
var pipeline = mlContext.Transforms.Concatenate(
    "Features", nameof(DatiPannello.OreDiSole), nameof(DatiPannello.Temperatura), nameof(DatiPannello.Inclinazione))
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Guadagno", maximumNumberOfIterations: 500));

#region Predizione
//Guadagno = a * OreDiSole + b * Temperatura + c * Inclinazione + d
//var model = pipeline.Fit(trainingData);

//var predEngine = mlContext.Model.CreatePredictionEngine<DatiPannello, PrevisioneGuadagno>(model);

//var input = new DatiPannello() { OreDiSole = 7.5f, Inclinazione = 28, Temperatura = 31 };
//var pred = predEngine.Predict(input);

//Console.WriteLine(pred.Score);
#endregion

//PRESTAZIONI

var model = pipeline.Fit(trainingData);
var predictionsTraining = model.Transform(trainingData);

var metricheTraining = mlContext.Regression.Evaluate(predictionsTraining, labelColumnName: "Guadagno");

Console.WriteLine($"R^2: {metricheTraining.RSquared}");             // Vicino a 1 è meglio
Console.WriteLine($"MAE: {metricheTraining.MeanAbsoluteError}");    //Mean Absolute Error
Console.WriteLine($"RMSE: {metricheTraining.RootMeanSquaredError}");    //Serve per vedere se ci sono OUTLIERS

var dataSetTest = new List<DatiPannello>
{
    new DatiPannello(){ OreDiSole = 7, Temperatura = 31, Inclinazione = 23, Guadagno = 200 },
    new DatiPannello(){ OreDiSole = 11, Temperatura = 30, Inclinazione = 30, Guadagno = 220 },
};
var testSet = mlContext.Data.LoadFromEnumerable(dataSetTest);

var predictions = model.Transform(testSet);

var metriche = mlContext.Regression.Evaluate(predictions, labelColumnName: "Guadagno");

Console.WriteLine($"R^2: {metriche.RSquared}");             // Vicino a 1 è meglio
Console.WriteLine($"MAE: {metriche.MeanAbsoluteError}");    //Mean Absolute Error
Console.WriteLine($"RMSE: {metriche.RootMeanSquaredError}");    //Serve per vedere se ci sono OUTLIERS





// Valore di R^2
// 1 = Perfezione, spiego perfettamente tutti i dati che ho nel Training Set con il modello
// 0 = Non riesco a spiegare il modello, se faccio una media di tutte le variabili potrebbe esser meglio
// < 0 = Il modello è peggiore di una quasiasi previsione media


#region Classes

public class DatiPannello
{
    public float OreDiSole { get; set; }
    public float Temperatura { get; set; }
    public float Inclinazione { get; set; }
    public float Guadagno { get; set; }         //Var DIP
}

public class PrevisioneGuadagno
{
    public float Score { get; set; }            //Var DIP - REGRESSIONE LINEARE, valore previsto
}

#endregion