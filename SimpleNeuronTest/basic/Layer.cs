namespace SimpleNeuronTest.basic;

public class Layer
{
    public Neuron[] Neurons { get; }

    public Layer(int neuronCount, int inputCountPerNeuron, Random rng)
    {
        Neurons = new Neuron[neuronCount];
        for (var i = 0; i < neuronCount; i++)
            Neurons[i] = new Neuron(inputCountPerNeuron, rng);
    }

    public double[] Forward(double[] inputs)
    {
        var outputs = new double[Neurons.Length];
        for (var i = 0; i < Neurons.Length; i++)
            outputs[i] = Neurons[i].Forward(inputs);
        return outputs;
    }
}