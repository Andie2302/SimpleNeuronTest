namespace SimpleNeuronTest;

public class LearnedXor
{
    // Die gelernten Konstanten aus deinem 10-Millionen-Lauf
    private const double H0W0 = 7.48399429;
    private const double H0W1 = -7.63352400;
    private const double H0B  = -3.96533977;

    private const double H1W0 = 7.64419262;
    private const double H1W1 = -7.43677222;
    private const double H1B  = 3.71915677;

    private const double OutW0 = 14.77497287;
    private const double OutW1 = -14.49729010;
    private const double OutB  = 7.03850020;

    // Mathematische Hilfsfunktion
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public double Predict(double x0, double x1)
    {
        // Hidden Layer Inferenz (ohne Schleifen, direkt berechnet)
        double h0 = Sigmoid(x0 * H0W0 + x1 * H0W1 + H0B);
        double h1 = Sigmoid(x0 * H1W0 + x1 * H1W1 + H1B);

        // Output Layer Inferenz
        return Sigmoid(h0 * OutW0 + h1 * OutW1 + OutB);
    }
}