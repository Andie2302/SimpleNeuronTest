using SimpleNeuronTest;

var net = new SimpleNetwork();

// Trainingsdaten für eine UND-Verknüpfung
double[][] inputs =
[
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];

//logisches & ist das Ziel
//double[] targets = { 0, 0, 0, 1 };

//XOR ist das Ziel...
double[] targets = [0, 1, 1, 0];

Console.WriteLine("Training startet...");

for (var epoch = 0; epoch < 10000; epoch++)
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
    Console.WriteLine($"Input: {input[0]}, {input[1]} -> Output: {result:F4}");
}