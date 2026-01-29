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

        // 2. Fehler am Ausgang berechnen (Soll - Ist)
        double error = target - actual;

        // 3. Backward Pass
        // Zuerst das Output-Neuron trainieren, es liefert den Fehler für den Hidden-Layer zurück
        double hiddenErrorSignal = OutputNeuron.Train(error, learningRate);

        // Dann den Hidden-Layer trainieren
        foreach (var neuron in HiddenLayer)
        {
            neuron.Train(hiddenErrorSignal, learningRate);
        }
    }
}