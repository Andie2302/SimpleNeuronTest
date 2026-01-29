using SimpleNeuronTest.basic;



TrainingTest.RunXorTraining();


Console.WriteLine("Test der gelernten XOR-Klasse:");
Console.WriteLine($"0, 0 -> {LearnedXor.Predict(0, 0):F4}");
Console.WriteLine($"0, 1 -> {LearnedXor.Predict(0, 1):F4}");
Console.WriteLine($"1, 0 -> {LearnedXor.Predict(1, 0):F4}");
Console.WriteLine($"1, 1 -> {LearnedXor.Predict(1, 1):F4}");
