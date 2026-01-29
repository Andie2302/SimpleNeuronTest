namespace SimpleNeuronTest;

public class Neuron
{
    public double[] Weights;
    public double Bias;
    
    // Speicher für den Rückweg
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
        // Wir speichern die Inputs und den Output für das Training
        _lastInputs = (double[])inputs.Clone();
        var sum = Bias;
        for (var i = 0; i < Weights.Length; i++)
        {
            sum += inputs[i] * Weights[i];
        }
        LastOutput = Sigmoid(sum);
        return LastOutput;
    }

    // Die mathematische Korrektur
    public double Train(double errorSignal, double learningRate)
    {
        // Der Gradient: Wie stark ändert sich der Output bei kleiner Änderung der Summe?
        // Ableitung von Sigmoid: f'(x) = f(x) * (1 - f(x))
        double delta = errorSignal * LastOutput * (1.0 - LastOutput);

        // Wir berechnen den Fehler für den vorherigen Layer, BEVOR wir die Gewichte ändern
        double errorForPreviousLayer = 0;
        for (int i = 0; i < Weights.Length; i++)
        {
            errorForPreviousLayer += Weights[i] * delta;
            
            // Gewicht anpassen: Lernrate * Gradient * Input
            Weights[i] += learningRate * delta * _lastInputs[i];
        }

        Bias += learningRate * delta;
        
        return errorForPreviousLayer;
    }
}