//Creare un modello che predica il guadagno di un pannello solare in relazione alle ore di sole che si applicano durante la giornata

using Microsoft.ML;


//Creazione contesto
var mlContext = new MLContext();
var path = "modello_giovanni.zip";
ITransformer modello;

if (File.Exists(path))
{
    #region Caricamento modello
    using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
    {
        modello = mlContext.Model.Load(fs, out var schemaModello);
    }
    Console.WriteLine("Modello caricato esistente!");
    #endregion
}
else
{
    var dataset = new[]
    {
        new DatiSole(){ OreSole = 1.0f, Guadagno = 100 },
        new DatiSole(){ OreSole = 2.0f, Guadagno = 150 },
        new DatiSole(){ OreSole = 3.0f, Guadagno = 200 },
        new DatiSole(){ OreSole = 4.0f, Guadagno = 250 },
        new DatiSole(){ OreSole = 5.0f, Guadagno = 300 }       //Prev 6
    };

    //Converto i dati in DataView
    var trainingSet = mlContext.Data.LoadFromEnumerable(dataset);

    //Creo la pipeline di regressione
    var pipeline = mlContext.Transforms.Concatenate("Features", nameof(DatiSole.OreSole))
        .Append(mlContext.Regression.Trainers.Sdca("Guadagno"));

    //Addestro il modello
    modello = pipeline.Fit(trainingSet);

    #region Salvataggio modello
    using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
    {
        mlContext.Model.Save(modello, trainingSet.Schema, fs);
    }
    Console.WriteLine("Modello salvato!");
    #endregion
}

//Creazione del Prediction Engine
var predEngi = mlContext.Model
.CreatePredictionEngine<DatiSole, PrevisioneGuadagno>(modello);

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