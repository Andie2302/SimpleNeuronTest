

namespace SimpleNeuronTest;

public class LearnedXor
{
    private readonly record struct NeuronWeights(double W0, double W1, double Bias);

    private static readonly NeuronWeights Hidden0 = new(7.48399429, -7.63352400, -3.96533977);
    private static readonly NeuronWeights Hidden1 = new(7.64419262, -7.43677222, 3.71915677);
    private static readonly NeuronWeights Output = new(14.77497287, -14.49729010, 7.03850020);

    private static double Sigmoid01(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Forward(in NeuronWeights neuron, double x0, double x1)
    {
        var z = (x0 * neuron.W0) + (x1 * neuron.W1) + neuron.Bias;
        return Sigmoid01(z);
    }

    public static double Predict(double x0, double x1)
    {
        var h0 = Forward(Hidden0, x0, x1);
        var h1 = Forward(Hidden1, x0, x1);

        return Forward(Output, h0, h1);
    }
}