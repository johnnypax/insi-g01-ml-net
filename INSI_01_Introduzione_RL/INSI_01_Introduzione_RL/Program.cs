//Prevedere il tempo di cottura della pizza nel forno, utilizzando ML.NET

//Training Dataset
using Microsoft.ML;
using System.Diagnostics;

var dataset = new[]
{
    new PizzaInfo(){ TemperaturaForno = 180, TempoCottura = 15},
    new PizzaInfo(){ TemperaturaForno = 200, TempoCottura = 12},
    new PizzaInfo(){ TemperaturaForno = 220, TempoCottura = 10},
    new PizzaInfo(){ TemperaturaForno = 250, TempoCottura = 8}
};

//ATTENZIONE: Serve sempre ML Context
var mlContext = new MLContext();

//Creazione del DataView
var trainingData = mlContext.Data.LoadFromEnumerable(dataset);

//Definisco la pipeline di addestramento
//STEP 1: Concatena a Features tutte le variabili INDIPENDENTI
//STEP 2: Fai l'append ad ogni dato presente in Features dei dati di TempoCottura e configurali per il training
var pipeline = mlContext
    .Transforms.Concatenate("Features", nameof(PizzaInfo.TemperaturaForno))
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "TempoCottura"));

//Costruzione del modello
var modelTimer = Stopwatch.StartNew();
var model = pipeline.Fit(trainingData);
modelTimer.Stop();
Console.WriteLine($"Tempo impiegato per la creazione del modello: {modelTimer.Elapsed.Milliseconds} ms");

//Inizializza un Prediction Engine
var predTimer = Stopwatch.StartNew();
var predEngine = mlContext.Model.CreatePredictionEngine<PizzaInfo, PredizioneCottura>(model);
predTimer.Stop();
Console.WriteLine($"Tempo impiegato per la predizione: {predTimer.Elapsed.Milliseconds} ms");

var testCottura = new PizzaInfo() { TemperaturaForno = 300 };
PredizioneCottura predizione = predEngine.Predict(testCottura);
Console.WriteLine($"Tempo di cottura predetto: {predizione.Score}");

#region Classes
class PizzaInfo
{                                                   //Lavoro con i float per ML.NET
    public float TemperaturaForno { get; set; }     //Variabile INDIPENDENTE (sensore)
    public float TempoCottura { get; set; }         //Variabile DIPENDENTE
}

class PredizioneCottura
{
    public float Score { get; set; }    //temperatura predetta, variabile DIPENDENTE
}

#endregion