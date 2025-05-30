//Creare un sistema che preveda un intervento tecnico tenendo conto di due fattori: Temperatura del motore elettrico e ore di running

/*
 * OBIETTIVO:
 * Necesita manutenzione? SI
 * Probabilità: 30%
 */

using Microsoft.ML;
using Microsoft.ML.Data;

var dataSet = new List<DatiMotore>
{
    new DatiMotore { Temperatura = 70, OreDiLavoro = 100, NecessitaMan = false },
    new DatiMotore { Temperatura = 80, OreDiLavoro = 200, NecessitaMan = false },
    new DatiMotore { Temperatura = 95, OreDiLavoro = 300, NecessitaMan = true },
    new DatiMotore { Temperatura = 100, OreDiLavoro = 400, NecessitaMan = true },
};

var mlContext = new MLContext();

var trainingData = mlContext.Data.LoadFromEnumerable(dataSet);

//Features = [Temperatura, OreDiLavoro]
var pipeline = mlContext.Transforms
    .Concatenate("Features", nameof(DatiMotore.Temperatura), nameof(DatiMotore.OreDiLavoro))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("NecessitaMan", maximumNumberOfIterations: 500));

var model = pipeline.Fit(trainingData);

var predEngine = mlContext.Model.CreatePredictionEngine<DatiMotore, PredizioneMan>(model);

//Test di input

var pred = predEngine.Predict(new DatiMotore() { Temperatura = 85, OreDiLavoro = 240 });

var probabilita = (1 / (1 + Math.Exp(-pred.Score)) ) * 100;  //Trasformo il valore grezzo in Probabilità

Console.WriteLine($"Bisogno di manutenzione? {pred.Prediction}");
Console.WriteLine($"Probabilita: {probabilita} %");

#region Classi

public class DatiMotore()
{
    public float Temperatura { get; set; }      //Variabile IND
    public float OreDiLavoro { get; set; }      //Variabile IND
    public bool  NecessitaMan { get; set; }     //Variabile DIP
}

public class PredizioneMan()
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }        //Variabile DIP

    public float Score { get; set; }            //Probabilità grezza (LOGIT)
};

#endregion