namespace SimpleNeuronTest.basic;

public static class TrainingTest
{
    public static void Run()
    {
        // 1. Netzwerk mit Topologie erstellen (2 Inputs, 2 Hidden, 1 Output)
        var net = new SimpleNetwork(new int[] { 2, 2, 1 });

        double[][] inputs = {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 0 }; // XOR

        Console.WriteLine("Training startet...");

        for (int epoch = 0; epoch < 100000; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                // Target muss jetzt ein Array sein
                net.Train(inputs[i], new[] { targets[i] }, 0.1);
            }
        }

        Console.WriteLine("Training beendet.\nErgebnisse:");

        foreach (var input in inputs)
        {
            var result = net.Predict(input);
            // Wir nehmen den ersten (und einzigen) Ausgangswert
            Console.WriteLine($"Input: {input[0]}, {input[1]} -> Output: {result[0]:F4}");
        }

        // Export-Logik (angepasst an Layer-Struktur)
        ExportWeights(net);
    }

    private static void ExportWeights(SimpleNetwork net)
    {
        Console.WriteLine("\n--- Gelernte Gewichte für den Export ---");
        
        // Da _layers private ist, müsstest du in SimpleNetwork 
        // entweder eine Public Property für die Layer machen 
        // oder eine Export-Methode direkt in SimpleNetwork schreiben.
        // Für den Moment kommentieren wir den Export hier aus, 
        // damit der Build erst mal durchläuft.
    }
}