//Seguendo i valori di temperatura precedenti, voglio decidere (rispetto alle mie decisioni precedenti) se andare o meno al mare.

using Microsoft.ML;
using Microsoft.ML.Data;

var dataSet = new[]
{
    new ViaggioSpiaggia(){ Temperatura = 18, VaiAlMare = false},
    new ViaggioSpiaggia(){ Temperatura = 19, VaiAlMare = false},
    new ViaggioSpiaggia(){ Temperatura = 20, VaiAlMare = false},
    new ViaggioSpiaggia(){ Temperatura = 21, VaiAlMare = false},
    new ViaggioSpiaggia(){ Temperatura = 22, VaiAlMare = true},
    new ViaggioSpiaggia(){ Temperatura = 23, VaiAlMare = true},
};

var mlContext = new MLContext();

//Trasformazione dei dati in DataView
var traingData = mlContext.Data.LoadFromEnumerable(dataSet);

var pipeline = mlContext.Transforms.Concatenate("Features", nameof(ViaggioSpiaggia.Temperatura))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("VaiAlMare"));

#region Classi

public class ViaggioSpiaggia()
{
    public float Temperatura { get; set; }

    public bool VaiAlMare { get; set; }         //Variabile DIP
}

public class PrevisioneViaggio
{
    [ColumnName("PredictionLabel")]
    public bool Prediction { get; set; }         //Variabile DIP 
}

#endregion