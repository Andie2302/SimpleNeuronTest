namespace SimpleNeuronTest.basic;

public static class CircleTrainingTest
{
    public static void Run()
    {
        // Dein tiefes 2-10-10-10-1 Netzwerk
        var net = new SimpleNetwork([2, 10, 10, 10, 1]);
        var rng = new Random(42);

        Console.WriteLine("Training: Kreis-Erkennung startet...");

        // Wir trainieren mit 50.000 zufälligen Punkten
        for (int i = 0; i < 500_000; i++)
        {
            double x = rng.NextDouble(); // 0.0 bis 1.0
            double y = rng.NextDouble();
            
            // Abstand zum Mittelpunkt (0.5, 0.5) berechnen
            double dist = Math.Sqrt(Math.Pow(x - 0.5, 2) + Math.Pow(y - 0.5, 2));
            
            // Liegt der Punkt im Kreis (Radius 0.3)?
            double target = dist < 0.3 ? 1.0 : 0.0;

            net.Train([x, y], [target], 0.05);

            if (i % 10000 == 0) Console.WriteLine($"Fortschritt: {i} Punkte verarbeitet...");
        }

        Console.WriteLine("Training beendet. Teste Quadranten...");

        // Teste ein paar markante Punkte
        double[][] testPoints = [
            [0.5, 0.5], // Volltreffer Mitte (Soll 1)
            [0.5, 0.7], // Innerhalb (Soll 1)
            [0.1, 0.1], // Weit außerhalb (Soll 0)
            [0.8, 0.8]  // Weit außerhalb (Soll 0)
        ];

        foreach (var p in testPoints)
        {
            var res = net.Predict(p);
            Console.WriteLine($"Punkt ({p[0]}|{p[1]}) -> Wahrscheinlichkeit für Kreis: {res[0]:P2}");
        }
    }
    
    public static void Visualize(SimpleNetwork net)
    {
        Console.WriteLine("\n--- Visualisierung des gelernten Kreises ---");
        for (int y = 0; y <= 20; y++)
        {
            for (int x = 0; x <= 40; x++)
            {
                // Wir scannen das Feld von 0.0 bis 1.0
                double xVal = x / 40.0;
                double yVal = y / 20.0;
            
                var res = net.Predict([xVal, yVal]);
            
                // Wenn Wert > 0.5, zeichne ein Zeichen, sonst Leerzeichen
                Console.Write(res[0] > 0.5 ? "#" : ".");
            }
            Console.WriteLine();
        }
    }
    
}