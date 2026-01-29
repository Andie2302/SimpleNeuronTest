namespace SimpleNeuronTest;

public class Neuron
{
    public double[] Weights;
    public double Bias;

    public Neuron(int inputCount, Random rng)
    {
        Weights = new double[inputCount];
        // Initialisierung mit kleinen Zufallswerten (wie in der KI üblich)
        for (var i = 0; i < inputCount; i++)
            Weights[i] = rng.NextDouble() * 2 - 1; 
        
        Bias = rng.NextDouble() * 2 - 1;
    }

    // Die Aktivierungsfunktion (Sigmoid)
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double Forward(double[] inputs)
    {
        var sum = Bias;
        for (var i = 0; i < Weights.Length; i++)
        {
            sum += inputs[i] * Weights[i];
        }
        return Sigmoid(sum);
    }
}