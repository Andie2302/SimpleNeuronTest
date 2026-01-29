namespace SimpleNeuronTest.basic;

public class TrainingTest
{
    public static void RunXorTraining()
    {
        // 2 Inputs, 2 Hidden-Neuronen, 1 Output
        var net = new SimpleNetwork([2, 10,10,10, 1]);

        double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[][] targets = [[0], [1], [1], [0]]; // XOR Targets als Arrays

        Console.WriteLine("Training startet...");

        for (var epoch = 0; epoch < 100_000; epoch++)
        {
            for (var i = 0; i < inputs.Length; i++)
            {
                net.Train(inputs[i], targets[i], 0.1);
            }
        }

        Console.WriteLine("Training beendet.\nErgebnisse:");
        foreach (var input in inputs)
        {
            var result = net.Predict(input);
            Console.WriteLine($"Input: {input[0]}, {input[1]} -> Output: {result[0]:F4}");
        }

        ExportWeights(net);
    }

    private static void ExportWeights(SimpleNetwork net)
    {
        Console.WriteLine("\n--- Modularer Export ---");
        for (int l = 0; l < net.Layers.Count; l++)
        {
            Console.WriteLine($"// Layer {l}");
            var layer = net.Layers[l];
            for (int n = 0; n < layer.Neurons.Length; n++)
            {
                var neuron = layer.Neurons[n];
                Console.WriteLine($"// Neuron {n} weights: {string.Join(", ", neuron.Weights.Select(w => w.ToString("F8")))}");
                Console.WriteLine($"// Neuron {n} bias: {neuron.Bias:F8}");
            }
        }
    }
}