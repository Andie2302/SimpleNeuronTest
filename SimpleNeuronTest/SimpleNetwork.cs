using SimpleNeuronTest;

public class SimpleNetwork
{
    public Neuron[] HiddenLayer;
    public Neuron OutputNeuron;

    public SimpleNetwork()
    {
        var rng = new Random(42); // Seed für Vorhersehbarkeit

        // 2 Neuronen im Hidden Layer, jedes hat 2 Eingänge
        HiddenLayer = new Neuron[] { new Neuron(2, rng), new Neuron(2, rng) };

        // 1 Output Neuron, es hat 2 Eingänge (von den 2 Hidden Neuronen)
        OutputNeuron = new Neuron(2, rng);
    }

    public double Predict(double[] inputs)
    {
        // 1. Forward Pass durch den Hidden Layer
        var h1 = HiddenLayer[0].Forward(inputs);
        var h2 = HiddenLayer[1].Forward(inputs);

        // 2. Die Ergebnisse der Hidden Neuronen sammeln
        var hiddenOutputs = new double[] { h1, h2 };

        // 3. Forward Pass durch das Output Neuron
        return OutputNeuron.Forward(hiddenOutputs);
    }
}