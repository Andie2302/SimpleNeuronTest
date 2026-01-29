namespace SimpleNeuronTest;

public class Neuron
{
    public readonly double[] Weights;
    public double Bias;
    private double[] _lastInputs = null!;
    public double LastOutput { get; private set; }

    public Neuron(int inputCount, Random rng)
    {
        Weights = new double[inputCount];
        for (var i = 0; i < inputCount; i++)
            Weights[i] = rng.NextDouble() * 2 - 1; 
        
        Bias = rng.NextDouble() * 2 - 1;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double Forward(double[] inputs)
    {
        _lastInputs = (double[])inputs.Clone();
        var sum = Bias;
        for (var i = 0; i < Weights.Length; i++)
            sum += inputs[i] * Weights[i];
        
        LastOutput = Sigmoid(sum);
        return LastOutput;
    }

    // SCHRITT 1: Berechne nur das "Delta" (den lokalen Gradienten)
    public double CalculateDelta(double errorSignal)
    {
        // f'(x) = f(x) * (1 - f(x))
        return errorSignal * LastOutput * (1.0 - LastOutput);
    }

    // SCHRITT 2: Wende die Korrektur an
    public void UpdateWeights(double delta, double learningRate)
    {
        for (var i = 0; i < Weights.Length; i++)
        {
            Weights[i] += learningRate * delta * _lastInputs[i];
        }
        Bias += learningRate * delta;
    }
}