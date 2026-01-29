namespace SimpleNeuronTest.basic;

public class Neuron
{
    private const double InitialWeightRange = 1.0;

    private readonly double[] _weights;
    private double[] _lastInputs = Array.Empty<double>();

    public double[] Weights => _weights;
    public double Bias { get; private set; }
    public double LastOutput { get; private set; }

    public Neuron(int inputCount, Random rng)
    {
        if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
        ArgumentNullException.ThrowIfNull(rng);

        _weights = new double[inputCount];
        for (var i = 0; i < inputCount; i++)
            _weights[i] = NextWeight(rng);

        Bias = NextWeight(rng);
    }

    private static double NextWeight(Random rng) => (rng.NextDouble() * 2 * InitialWeightRange) - InitialWeightRange;

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double Forward(double[] inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Length != _weights.Length)
            throw new ArgumentException($"Expected {nameof(inputs)} length {_weights.Length}, but got {inputs.Length}.", nameof(inputs));

        _lastInputs = (double[])inputs.Clone();

        var activation = Bias + _weights.Select((t, i) => inputs[i] * t).Sum();

        LastOutput = Sigmoid(activation);
        return LastOutput;
    }

    public double CalculateDelta(double errorSignal)
        => errorSignal * LastOutput * (1.0 - LastOutput);

    public void UpdateWeights(double delta, double learningRate)
    {
        if (_lastInputs.Length != _weights.Length)
            throw new InvalidOperationException($"{nameof(Forward)} must be called before {nameof(UpdateWeights)}.");

        for (var i = 0; i < _weights.Length; i++)
            _weights[i] += learningRate * delta * _lastInputs[i];

        Bias += learningRate * delta;
    }
}