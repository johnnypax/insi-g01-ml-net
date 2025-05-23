//Creare un modello che predica il guadagno di un pannello solare in relazione alle ore di sole che si applicano durante la giornata

using Microsoft.ML;

var dataset = new[]
{
    new DatiSole(){ OreSole = 1, Guadagno = 100 },
    new DatiSole(){ OreSole = 2, Guadagno = 150 },
    new DatiSole(){ OreSole = 3, Guadagno = 200 },
    new DatiSole(){ OreSole = 4, Guadagno = 250 },
    new DatiSole(){ OreSole = 5, Guadagno = 300 }       //Prev 6
};

//Creazione contesto
var mlContext = new MLContext();

//Converto i dati in DataView
var trainingSet = mlContext.Data.LoadFromEnumerable(dataset);

//Creo la pipeline di regressione
var pipeline = mlContext.Transforms.Concatenate("Features", nameof(DatiSole.OreSole))
    .Append(mlContext.Regression.Trainers.Sdca("Guadagno", maximumNumberOfIterations: 300));

//Addestro il modello
var modello = pipeline.Fit(trainingSet);

//Creazione del Prediction Engine
var predEngi = mlContext.Model
.CreatePredictionEngine<DatiSole, PrevisioneGuadagno>(modello);

//Predizione
var input = new DatiSole() { OreSole = 6 };
PrevisioneGuadagno predizione = predEngi.Predict(input);

Console.WriteLine(predizione.Score);

//RENDERING

var plot = new ScottPlot.Plot();

//FASE 1: inserimento dei dati originali
double[] datiX = dataset.Select(g => (double)g.OreSole).ToArray();
double[] datiY = dataset.Select(g => (double)g.Guadagno).ToArray();

var scatter = plot.Add.Scatter(datiX, datiY);
scatter.LegendText = "Dati originali";
scatter.MarkerSize = 15;

//FASE 2: Creazione della linea di Regressione
double[] rlX = Enumerable.Range(-2, 7).Select(i => (double)i).ToArray();
double[] rlY = rlX.Select(
    val => (double)predEngi.Predict(new DatiSole() { OreSole = (float)val }).Score)
    .ToArray();

var regressioneScatter = plot.Add.ScatterLine(rlX, rlY);
regressioneScatter.LegendText = "Linea di Regressione";

plot.SavePng("guadagno.png", 1024, 1024);

#region Classi
public class DatiSole()
{
    public float OreSole { get; set; }      // Variabile IND
    public float Guadagno { get; set; }     // Variabile DIP
}

public class PrevisioneGuadagno
{
    public float Score { get; set; }        // Variabile DIP
}

#endregion