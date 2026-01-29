// See https://aka.ms/new-console-template for more information

Console.WriteLine("Hello, World!");

var net = new SimpleNetwork();
double[] sensorDaten = { 0.5, -0.2 }; // Beispiel-Inputs

var result = net.Predict(sensorDaten);
Console.WriteLine($"Netzwerk-Ausgang: {result:F4}");