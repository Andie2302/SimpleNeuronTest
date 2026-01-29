using SimpleNeuronTest;

/*
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

for (var epoch = 0; epoch < 10_000_000; epoch++)
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


Console.WriteLine("\n--- Gelernte Gewichte für den Export ---");

// Hidden Layer Gewichte
for (int i = 0; i < net.HiddenLayer.Length; i++)
{
    var n = net.HiddenLayer[i];
    Console.WriteLine($"// Hidden Neuron {i}");
    Console.WriteLine($"double h{i}_weights[] = {{ {string.Join(", ", n.Weights.Select(w => w.ToString("F8")))} }};");
    Console.WriteLine($"double h{i}_bias = {n.Bias.ToString("F8")};");
}

// Output Neuron Gewichte
var outN = net.OutputNeuron;
Console.WriteLine("// Output Neuron");
Console.WriteLine($"double out_weights[] = {{ {string.Join(", ", outN.Weights.Select(w => w.ToString("F8")))} }};");
Console.WriteLine($"double out_bias = {outN.Bias.ToString("F8")};");

//*/


var xorGate = new LearnedXor();

Console.WriteLine("Test der gelernten XOR-Klasse:");
Console.WriteLine($"0, 0 -> {xorGate.Predict(0, 0):F4}");
Console.WriteLine($"0, 1 -> {xorGate.Predict(0, 1):F4}");
Console.WriteLine($"1, 0 -> {xorGate.Predict(1, 0):F4}");
Console.WriteLine($"1, 1 -> {xorGate.Predict(1, 1):F4}");