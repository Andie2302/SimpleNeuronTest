using SimpleNeuronTest;

public class SimpleNetwork
{
    public Neuron[] HiddenLayer;
    public Neuron OutputNeuron;

    public SimpleNetwork()
    {
        var rng = new Random(42);
        HiddenLayer = new Neuron[] { new Neuron(2, rng), new Neuron(2, rng) };
        OutputNeuron = new Neuron(2, rng);
    }

    public double Predict(double[] inputs)
    {
        var h1 = HiddenLayer[0].Forward(inputs);
        var h2 = HiddenLayer[1].Forward(inputs);
        return OutputNeuron.Forward(new double[] { h1, h2 });
    }

    public void Train(double[] inputs, double target, double learningRate)
    {
        // 1. Forward Pass
        double actual = Predict(inputs);
        double error = target - actual;

        // 2. Deltas berechnen (Rückwärts)
        // Wie viel Fehler hat das Output-Neuron?
        double outputDelta = OutputNeuron.CalculateDelta(error);

        // Wie viel Fehler hat jedes einzelne Hidden-Neuron?
        // Hier ist die Korrektur: Wir nutzen das Gewicht der jeweiligen Verbindung!
        double[] hiddenDeltas = new double[HiddenLayer.Length];
        for (int i = 0; i < HiddenLayer.Length; i++)
        {
            // Der Fehler für Hidden-Neuron 'i' ist: 
            // Delta des Outputs * Gewicht der Verbindung von Hidden 'i' zum Output
            double errorSignalForHidden = outputDelta * OutputNeuron.Weights[i];
            hiddenDeltas[i] = HiddenLayer[i].CalculateDelta(errorSignalForHidden);
        }

        // 3. Gewichte anpassen (jetzt erst!)
        OutputNeuron.UpdateWeights(outputDelta, learningRate);
        for (int i = 0; i < HiddenLayer.Length; i++)
        {
            HiddenLayer[i].UpdateWeights(hiddenDeltas[i], learningRate);
        }
    }
}