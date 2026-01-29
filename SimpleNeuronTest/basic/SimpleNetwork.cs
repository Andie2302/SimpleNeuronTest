using SimpleNeuronTest.basic;

namespace SimpleNeuronTest;

public class SimpleNetwork
{
    private readonly Layer[] _layers;

    public SimpleNetwork(int[] topology, int seed = 42)
    {
        var rng = new Random(seed);
        _layers = new Layer[topology.Length - 1];

        // Wir erstellen die Layer basierend auf der Topologie
        // topology[0] ist der Input-Count, daher starten wir bei i=1
        for (var i = 0; i < _layers.Length; i++)
        {
            // Ein Layer hat topology[i+1] Neuronen, 
            // jedes mit topology[i] Eingängen
            _layers[i] = new Layer(topology[i + 1], topology[i], rng);
        }
    }

    public double[] Predict(double[] inputs)
    {
        var currentSignals = inputs;
        foreach (var layer in _layers)
        {
            currentSignals = layer.Forward(currentSignals);
        }
        return currentSignals;
    }

    public void Train(double[] inputs, double[] targets, double learningRate)
    {
        // 1. Forward Pass
        var actual = Predict(inputs);

        // 2. Backward Pass: Fehler für den Output-Layer berechnen
        var lastLayer = _layers[^1];
        var deltas = new double[lastLayer.Neurons.Length];
        var previousLayerErrors = new double[lastLayer.Neurons[0].Weights.Length];

        for (var i = 0; i < lastLayer.Neurons.Length; i++)
        {
            var error = targets[i] - actual[i];
            deltas[i] = lastLayer.Neurons[i].CalculateDelta(error);
        }

        // Hier würde nun die Schleife rückwärts durch alle Layer laufen
        // Um es übersichtlich zu halten, fangen wir mit dem Update des Output-Layers an:
        for (var i = 0; i < lastLayer.Neurons.Length; i++)
        {
            lastLayer.Neurons[i].UpdateWeights(deltas[i], learningRate);
        }
        
        // Hinweis: Für echte tiefe Netze müssen wir die deltas 
        // Schicht für Schicht nach vorne durchreichen (Backprop).
    }
}