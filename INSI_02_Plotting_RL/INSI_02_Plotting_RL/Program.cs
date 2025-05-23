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
    .Append(mlContext.Regression.Trainers.Sdca("Guadagno"));

//Addestro il modello
var modello = pipeline.Fit(trainingSet);

#region Salvataggio modello
var path = "modello_giovanni.zip";
using(var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    mlContext.Model.Save(modello, trainingSet.Schema, fs);
}
#endregion

#region Caricamento modello
ITransformer modelloCaricato;
using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
{
    modelloCaricato = mlContext.Model.Load(fs, out var schemaModello);
}
#endregion

//Creazione del Prediction Engine
var predEngi = mlContext.Model
.CreatePredictionEngine<DatiSole, PrevisioneGuadagno>(modelloCaricato);

//Predizione
var input = new DatiSole() { OreSole = 6 };
PrevisioneGuadagno predizione = predEngi.Predict(input);

Console.WriteLine(predizione.Score);

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