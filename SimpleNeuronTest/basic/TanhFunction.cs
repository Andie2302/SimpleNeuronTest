namespace SimpleNeuronTest.basic;

public class TanhFunction : IActivationFunction
{
    public double Activate(double x) => Math.Tanh(x);
    public double Derivative(double output) => 1.0 - output * output;
}