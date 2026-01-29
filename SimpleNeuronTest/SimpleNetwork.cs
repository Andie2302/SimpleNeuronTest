namespace SimpleNeuronTest;

public class SimpleNetwork
{
    private const int InputCount = 2;
    private const int HiddenNeuronCount = 2;
    private const int DefaultSeed = 42;

    public Neuron[] HiddenLayer { get; }
    public Neuron OutputNeuron { get; }

    public SimpleNetwork()
    {
        var rng = new Random(DefaultSeed);

        HiddenLayer = new Neuron[HiddenNeuronCount];
        for (var i = 0; i < HiddenLayer.Length; i++)
            HiddenLayer[i] = new Neuron(InputCount, rng);

        OutputNeuron = new Neuron(HiddenNeuronCount, rng);
    }

    public double Predict(double[] inputs)
    {
        var hiddenOutputs = ForwardHidden(inputs);
        return OutputNeuron.Forward(hiddenOutputs);
    }

    public void Train(double[] inputs, double target, double learningRate)
    {
        var prediction = Predict(inputs);
        var error = target - prediction;

        var outputDelta = OutputNeuron.CalculateDelta(error);

        var hiddenDeltas = new double[HiddenLayer.Length];
        for (var i = 0; i < HiddenLayer.Length; i++)
        {
            var errorSignalForHidden = outputDelta * OutputNeuron.Weights[i];
            hiddenDeltas[i] = HiddenLayer[i].CalculateDelta(errorSignalForHidden);
        }

        OutputNeuron.UpdateWeights(outputDelta, learningRate);

        for (var i = 0; i < HiddenLayer.Length; i++)
            HiddenLayer[i].UpdateWeights(hiddenDeltas[i], learningRate);
    }

    private double[] ForwardHidden(double[] inputs)
    {
        var hiddenOutputs = new double[HiddenLayer.Length];
        for (var i = 0; i < HiddenLayer.Length; i++)
            hiddenOutputs[i] = HiddenLayer[i].Forward(inputs);

        return hiddenOutputs;
    }
}