using SimpleNeuronTest;

var net = new SimpleNetwork();

// Trainingsdaten für eine UND-Verknüpfung
double[][] inputs = {
    new double[] { 0, 0 },
    new double[] { 0, 1 },
    new double[] { 1, 0 },
    new double[] { 1, 1 }
};
double[] targets = { 0, 0, 0, 1 };

Console.WriteLine("Training startet...");

for (int epoch = 0; epoch < 10000; epoch++)
{
    for (int i = 0; i < inputs.Length; i++)
    {
        net.Train(inputs[i], targets[i], 0.1);
    }
}

Console.WriteLine("Training beendet.\nErgebnisse:");

foreach (var input in inputs)
{
    double result = net.Predict(input);
    Console.WriteLine($"Input: {input[0]}, {input[1]} -> Output: {result:F4}");
}