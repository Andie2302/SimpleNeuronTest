namespace SimpleNeuronTest.basic;

public class SimpleNetwork
{
    private readonly Layer[] _layers;
    public IReadOnlyList<Layer> Layers => _layers;

    public SimpleNetwork(int[] topology, int seed = 42)
    {
        var rng = new Random(seed);
        _layers = new Layer[topology.Length - 1];

        for (var i = 0; i < _layers.Length; i++)
        {
            // Architektur-Entscheidung:
            // Letzter Layer = Sigmoid, alle anderen = ReLU
            IActivationFunction func = (i == _layers.Length - 1) 
                ? new SigmoidFunction() 
                : new ReluFunction();
            
            _layers[i] = new Layer(topology[i + 1], topology[i], func, rng);
        }
    }

    // ... Predict bleibt gleich ...

    public void Train(double[] inputs, double[] targets, double learningRate)
    {
        // 1. Forward Pass (Speichert inputs in den Neuronen)
        // [Hinweis: Du müsstest die Inputs pro Layer zwischenspeichern für UpdateWeights]
        // ... (Backprop Logik wie bisher, aber mit layer.GetDerivative)
    }
}