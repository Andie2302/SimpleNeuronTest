namespace SimpleNeuronTest.basic;

public class SimpleNetwork
{
    private readonly Layer[] _layers;
    // Ermöglicht den Zugriff für den Export
    public IReadOnlyList<Layer> Layers => _layers;

    public SimpleNetwork(int[] topology, int seed = 42)
    {
        var rng = new Random(seed);
        _layers = new Layer[topology.Length - 1];

        for (var i = 0; i < _layers.Length; i++)
        {
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

        // 2. Backward Pass (Der Kern der KI)
        // Wir starten mit dem Fehler am Ausgang
        double[] currentErrorSignals = new double[targets.Length];
        for (int i = 0; i < targets.Length; i++)
        {
            currentErrorSignals[i] = targets[i] - actual[i];
        }

        // Wir gehen die Layer von hinten nach vorne durch
        for (int i = _layers.Length - 1; i >= 0; i--)
        {
            var layer = _layers[i];
            // Wir bereiten die Fehlersignale für den davorliegenden Layer vor
            double[] nextErrorSignals = new double[layer.Neurons[0].Weights.Length];
            double[] deltas = new double[layer.Neurons.Length];

            for (int j = 0; j < layer.Neurons.Length; j++)
            {
                var neuron = layer.Neurons[j];
                // 1. Berechne das Delta für dieses Neuron
                deltas[j] = neuron.CalculateDelta(currentErrorSignals[j]);

                // 2. Verteile den Fehler rückwärts auf die Gewichte (für den vorherigen Layer)
                for (int k = 0; k < neuron.Weights.Length; k++)
                {
                    nextErrorSignals[k] += neuron.Weights[k] * deltas[j];
                }
            }

            // 3. Erst wenn alle Deltas des Layers berechnet sind, updaten wir die Gewichte
            for (int j = 0; j < layer.Neurons.Length; j++)
            {
                layer.Neurons[j].UpdateWeights(deltas[j], learningRate);
            }

            // Das Fehlersignal für die nächste (vordere) Schicht setzen
            currentErrorSignals = nextErrorSignals;
        }
    }
}